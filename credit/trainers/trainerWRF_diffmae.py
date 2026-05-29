"""Trainer for WoFS conditional precip diffusion / DiffMAE inpainting."""

from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


def _choose_mixed_height_mode(conf: dict, device: torch.device) -> str:
    probs = torch.tensor(
            [
                float(conf.get("mixed_height_spatial_probability", 1.0)),
                float(conf.get("mixed_height_channel_probability", 0.0)),
                float(conf.get("mixed_height_height_probability", 1.0)),
            ],
        device=device,
        dtype=torch.float32,
    )
    if torch.any(probs < 0):
        raise ValueError("Mixed height mask probabilities must be non-negative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("At least one mixed height mask probability must be positive")
    return ["spatial_patch", "channel_patch", "height_patch"][int(torch.multinomial(probs / total, 1).item())]


class TrainerDiffMAE(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _move_batch(self, batch: dict) -> dict:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != "index"
        }

    def _condition_dict(self, batch: dict) -> Dict[str, torch.Tensor]:
        return {
            "background": batch["background"],
            "surface": batch["surface"],
            "forcing": batch["forcing"],
            "reflectivity": batch["reflectivity"],
        }

    def _sample_mask(self, batch_size: int, trainer_conf: dict, device: torch.device, validation: bool = False):
        model = self._unwrap_model()
        ratio_key = "val_mask_ratio" if validation else "precip_mask_ratio"
        ratio = trainer_conf.get(ratio_key, trainer_conf.get("precip_mask_ratio", 1.0))
        mode_key = "val_mask_mode" if validation else "precip_mask_mode"
        mode = str(trainer_conf.get(mode_key, trainer_conf.get("precip_mask_mode", "spatial_patch"))).strip().lower()

        if mode in {"spatial_patch", "cube_patch", "patch", "3d_patch"}:
            return model.random_precip_mask(batch_size, ratio, device)
        if mode in {"mixed", "mixed_height", "height_patch", "channel_patch"}:
            raise ValueError(
                f"Mask mode {mode!r} is no longer supported by WoFSDiffMAE; use 'cube_patch' for 3D precip patches."
            )
        raise ValueError(f"Unsupported precip mask mode: {mode!r}")

    def _snapshot_mask(self, batch_size: int, conf: dict, device: torch.device) -> torch.Tensor:
        snapshot_conf = conf["trainer"].get("denoise_snapshot", {})
        mask_conf = dict(conf.get("eval", {}))
        mask_conf.update(snapshot_conf.get("mask", {}))
        if "precip_mask_ratio" not in mask_conf and "mask_ratio" not in mask_conf:
            mask_conf["precip_mask_ratio"] = conf.get("eval", {}).get("precip_mask_ratio", 0.75)
        if "precip_mask_mode" not in mask_conf and "mask_mode" not in mask_conf:
            mask_conf["precip_mask_mode"] = conf.get("eval", {}).get("precip_mask_mode", "cube_patch")
        return self._sample_mask(batch_size, mask_conf, device, validation=False)

    @staticmethod
    def _plot_field(ax, field: torch.Tensor, title: str, cmap: str = "viridis") -> None:
        arr = field.detach().float().cpu().numpy()
        finite = np.isfinite(arr)
        if finite.any():
            vmax = np.nanpercentile(np.abs(arr[finite]), 99)
            vmax = float(vmax) if vmax > 0 else 1.0
            im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _save_denoise_figure(
        self,
        out_path: Path,
        target: torch.Tensor,
        noisy: torch.Tensor,
        pred_x0: torch.Tensor,
        precip_mask: torch.Tensor,
        sample: Optional[torch.Tensor] = None,
        channel: int = 0,
    ) -> None:
        logger.info("Denoise snapshot figure start: path=%s target=%s mask=%s sample=%s",
                    out_path, tuple(target.shape), tuple(precip_mask.shape), None if sample is None else tuple(sample.shape))
        t0 = time.perf_counter()
        model = self._unwrap_model()
        mask_img_full = model.expand_patch_mask(precip_mask[:1], target.shape[-2], target.shape[-1])
        mask_img = mask_img_full.amax(dim=1, keepdim=True)
        precip_mask_img = mask_img_full.repeat_interleave(model.level_variable_count, dim=1)
        logger.info("Denoise snapshot masks expanded: level_mask=%s precip_mask=%s elapsed=%.3fs",
                    tuple(mask_img_full.shape), tuple(precip_mask_img.shape), time.perf_counter() - t0)
        target_1 = target[0]
        noisy_1 = noisy[0]
        pred_1 = pred_x0[0]
        masked_abs_err = (pred_1 - target_1).abs() * precip_mask_img[0]

        summary_target = target_1.mean(dim=0)
        summary_noisy = noisy_1.mean(dim=0)
        summary_pred = pred_1.mean(dim=0)
        summary_err = masked_abs_err.mean(dim=0)

        ncols = 6 if sample is not None else 5
        t_fig = time.perf_counter()
        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.0), constrained_layout=True)

        self._plot_field(axes[0, 0], summary_target, "target mean")
        self._plot_field(axes[0, 1], summary_noisy, "noisy mean")
        self._plot_field(axes[0, 2], summary_pred, "pred x0 mean")
        self._plot_field(axes[0, 3], summary_err, "masked abs err mean", cmap="magma")
        self._plot_field(axes[0, 4], mask_img[0, 0], "mask", cmap="gray")

        ch = max(0, min(int(channel), target_1.shape[0] - 1))
        self._plot_field(axes[1, 0], target_1[ch], f"target ch{ch}")
        self._plot_field(axes[1, 1], noisy_1[ch], f"noisy ch{ch}")
        self._plot_field(axes[1, 2], pred_1[ch], f"pred x0 ch{ch}")
        ch_mask = precip_mask_img[0, ch]
        self._plot_field(axes[1, 3], (pred_1[ch] - target_1[ch]).abs() * ch_mask, f"masked err ch{ch}", cmap="magma")
        self._plot_field(axes[1, 4], ch_mask, f"mask ch{ch}", cmap="gray")

        if sample is not None:
            sample_1 = sample[0]
            self._plot_field(axes[0, 5], sample_1.mean(dim=0), "sample mean")
            self._plot_field(axes[1, 5], sample_1[ch], f"sample ch{ch}")

        logger.info("Denoise snapshot figure populated: path=%s elapsed=%.3fs", out_path, time.perf_counter() - t_fig)
        t_save = time.perf_counter()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        logger.info("Denoise snapshot figure saved: path=%s elapsed=%.3fs total=%.3fs",
                    out_path, time.perf_counter() - t_save, time.perf_counter() - t0)

    def _maybe_save_denoise_snapshot(self, epoch: int, step: int, conf: dict, batch: dict, losses: dict) -> None:
        snapshot_conf = conf["trainer"].get("denoise_snapshot", {})
        if not snapshot_conf or not snapshot_conf.get("enabled", False) or self.rank != 0:
            return
        every = int(snapshot_conf.get("every_steps", 0))
        every_epoch = bool(snapshot_conf.get("every_epoch", False))
        first_step = max(0, int(snapshot_conf.get("first_step", 1)))
        should_save_step = every > 0 and step >= first_step and step % every == 0
        should_save_epoch = every_epoch and step == first_step
        if not (should_save_step or should_save_epoch):
            return
        out_dir = Path(conf["save_loc"]) / snapshot_conf.get("dir", "denoise_snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        model = self._unwrap_model()
        logger.info(
            "Denoise snapshot start: epoch=%s step=%s out_dir=%s save_snapshot=%s save_figure=%s "
            "save_sampling_figure=%s sampling_timesteps=%s",
            epoch,
            step,
            out_dir,
            bool(snapshot_conf.get("save_snapshot", True)),
            bool(snapshot_conf.get("save_figure", True)),
            bool(snapshot_conf.get("save_sampling_figure", False)),
            int(snapshot_conf.get("sampling_timesteps", 8)),
        )
        t_total = time.perf_counter()
        with torch.no_grad():
            precip_mask = losses.get("precip_mask")
            if precip_mask is None:
                precip_mask = self._snapshot_mask(1, conf, batch["precip"].device)
            t_pred = time.perf_counter()
            logger.info("Denoise snapshot model_predictions start: x_t=%s t=%s mask=%s",
                        tuple(losses["x_t"][:1].shape), tuple(losses["t"][:1].shape), tuple(precip_mask[:1].shape))
            pred_noise, pred_x0 = model.model_predictions(
                losses["x_t"][:1],
                losses["t"][:1],
                self._condition_dict({k: v[:1] for k, v in batch.items()}),
                None if getattr(model, "pure_diffusion", False) else precip_mask[:1],
                precip_visible=None if getattr(model, "pure_diffusion", False) else batch["precip"][:1],
                compose_inpaint=not getattr(model, "pure_diffusion", False),
            )
            logger.info("Denoise snapshot model_predictions done: pred_noise=%s pred_x0=%s elapsed=%.3fs",
                        tuple(pred_noise.shape), tuple(pred_x0.shape), time.perf_counter() - t_pred)
            sample = None
            if bool(snapshot_conf.get("save_sampling_figure", False)):
                sampling_steps = int(snapshot_conf.get("sampling_timesteps", 8))
                sample_visible = batch["precip"][:1] if bool(snapshot_conf.get("sampling_use_visible_precip", True)) else None
                sampler = str(snapshot_conf.get("sampler", "ddim"))
                inpaint_mode = snapshot_conf.get("inpaint_mode", None)
                log_inpaint_mode = inpaint_mode if inpaint_mode is not None else getattr(model, "default_inpaint_mode", "masked_only")
                eta = snapshot_conf.get("ddim_sampling_eta", None)
                t_sample = time.perf_counter()
                logger.info(
                    "Denoise snapshot sampling start: sampler=%s sampling_timesteps=%s visible=%s inpaint_mode=%s",
                    sampler,
                    sampling_steps,
                    sample_visible is not None,
                    log_inpaint_mode,
                )
                sample = model.sample_precip(
                    self._condition_dict({k: v[:1] for k, v in batch.items()}),
                    precip_mask[:1],
                    precip_visible=sample_visible,
                    sampling_timesteps=sampling_steps,
                    eta=eta,
                    sampler=sampler,
                    repaint_jump_length=int(snapshot_conf.get("repaint_jump_length", 10)),
                    repaint_jump_n_sample=int(snapshot_conf.get("repaint_jump_n_sample", 10)),
                    inpaint_mode=inpaint_mode,
                )
                logger.info("Denoise snapshot sampling done: sample=%s elapsed=%.3fs",
                            tuple(sample.shape), time.perf_counter() - t_sample)
            if bool(snapshot_conf.get("save_snapshot", True)):
                t_save_tensor = time.perf_counter()
                tensor_path = out_dir / f"epoch{epoch:04d}_step{step:06d}.pt"
                logger.info("Denoise snapshot tensor save start: path=%s", tensor_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "x_t": losses["x_t"][:1].detach().cpu(),
                        "target_precip": batch["precip"][:1].detach().cpu(),
                        "pred_x0": pred_x0[:1].detach().cpu(),
                        "pred_noise": pred_noise[:1].detach().cpu(),
                        "precip_mask": precip_mask[:1].detach().cpu(),
                        "sample": None if sample is None else sample[:1].detach().cpu(),
                    },
                    tensor_path,
                )
                logger.info("Denoise snapshot tensor save done: path=%s elapsed=%.3fs",
                            tensor_path, time.perf_counter() - t_save_tensor)
            if bool(snapshot_conf.get("save_figure", True)):
                t_render = time.perf_counter()
                figure_path = out_dir / f"epoch{epoch:04d}_step{step:06d}.png"
                logger.info("Denoise snapshot render start: path=%s", figure_path)
                self._save_denoise_figure(
                    figure_path,
                    target=batch["precip"][:1],
                    noisy=losses["x_t"][:1],
                    pred_x0=pred_x0[:1],
                    precip_mask=precip_mask[:1],
                    sample=None if sample is None else sample[:1],
                    channel=int(snapshot_conf.get("channel", 0)),
                )
                logger.info("Denoise snapshot render done: path=%s elapsed=%.3fs",
                            figure_path, time.perf_counter() - t_render)
        logger.info("Denoise snapshot done: epoch=%s step=%s total=%.3fs", epoch, step, time.perf_counter() - t_total)

    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        trainer_conf = conf["trainer"]
        batches_per_epoch = trainer_conf["batches_per_epoch"]
        grad_accum_every = int(trainer_conf.get("grad_accum_every", 1))
        grad_max_norm = trainer_conf.get("grad_max_norm", None)
        amp = bool(trainer_conf.get("amp", True))
        distributed = trainer_conf.get("mode", "none") in ("fsdp", "ddp")

        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        if not isinstance(trainloader.dataset, IterableDataset):
            batches_per_epoch = batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)

        self.model.train()
        results_dict = defaultdict(list)
        dl_iter = iter(trainloader)
        optimizer.zero_grad()
        pbar = tqdm.tqdm(range(batches_per_epoch), total=batches_per_epoch, leave=True, disable=(self.rank > 0))

        for i in pbar:
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(trainloader)
                batch = next(dl_iter)
            batch = self._move_batch(batch)
            model = self._unwrap_model()
            pure_diffusion = bool(getattr(model, "pure_diffusion", False) or trainer_conf.get("pure_diffusion_training", False))
            precip_mask = None
            precip_visible = None
            if not pure_diffusion:
                precip_mask = self._sample_mask(batch["precip"].shape[0], trainer_conf, batch["precip"].device)
                precip_visible = batch["precip"] if bool(trainer_conf.get("visible_precip_conditioning", False)) else None

            with autocast(enabled=amp):
                losses = model.p_losses(
                    batch["precip"],
                    self._condition_dict(batch),
                    precip_mask,
                    precip_visible=precip_visible,
                )
                loss = losses["loss"]

            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite precip diffusion loss before backward: epoch=%s step=%s rank=%s loss=%s",
                    epoch,
                    i,
                    self.rank,
                    loss.detach().float().item(),
                )
                loss_t = torch.tensor([float("nan")], device=self.device)
                if distributed:
                    dist.all_reduce(loss_t, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_loss"].append(loss_t.item())
                results_dict["train_precip_diffusion_loss"].append(loss_t.item())
                optimizer.zero_grad(set_to_none=True)
                break

            scaler.scale(loss / grad_accum_every).backward()
            if (i + 1) % grad_accum_every == 0:
                if grad_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_max_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_t = torch.tensor([loss.item()], device=self.device)
            if distributed:
                dist.all_reduce(loss_t, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(loss_t.item())
            results_dict["train_precip_diffusion_loss"].append(loss_t.item())

            if precip_mask is not None:
                losses["precip_mask"] = precip_mask
            self._maybe_save_denoise_snapshot(epoch, i, conf, batch, losses)

            if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch:
                if (i + 1) % grad_accum_every == 0:
                    scheduler.step()

            if self.rank == 0:
                pbar.set_description(
                    "Epoch: {} train_loss: {:.6f} lr: {:.3e}".format(
                        epoch, np.mean(results_dict["train_loss"]), optimizer.param_groups[0]["lr"]
                    )
                )
            if not np.isfinite(np.mean(results_dict["train_loss"])):
                logger.warning("Non-finite precip diffusion loss encountered, stopping epoch early.")
                break

        pbar.close()
        torch.cuda.empty_cache()
        gc.collect()
        for key in ["acc", "mae", "forecast_len"]:
            results_dict[f"train_{key}"] = [0.0]
        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        trainer_conf = conf["trainer"]
        valid_batches = trainer_conf.get("valid_batches_per_epoch", 0)
        amp = bool(trainer_conf.get("amp", True))
        distributed = trainer_conf.get("mode", "none") in ("fsdp", "ddp")
        n_valid = valid_batches if 0 < valid_batches < len(valid_loader) else len(valid_loader)

        self.model.eval()
        results_dict = defaultdict(list)
        pbar = tqdm.tqdm(range(n_valid), total=n_valid, leave=True, disable=(self.rank > 0))
        with torch.no_grad():
            dl_iter = iter(valid_loader)
            for _ in pbar:
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    break
                batch = self._move_batch(batch)
                model = self._unwrap_model()
                pure_diffusion = bool(getattr(model, "pure_diffusion", False) or trainer_conf.get("pure_diffusion_training", False))
                precip_mask = None
                precip_visible = None
                if not pure_diffusion:
                    precip_mask = self._sample_mask(
                        batch["precip"].shape[0],
                        trainer_conf,
                        batch["precip"].device,
                        validation=True,
                    )
                    precip_visible = batch["precip"] if bool(trainer_conf.get("visible_precip_conditioning", False)) else None
                with autocast(enabled=amp):
                    losses = model.p_losses(
                        batch["precip"],
                        self._condition_dict(batch),
                        precip_mask,
                        precip_visible=precip_visible,
                    )
                    loss = losses["loss"]
                loss_t = torch.tensor([loss.item()], device=self.device)
                if distributed:
                    dist.all_reduce(loss_t, dist.ReduceOp.AVG, async_op=False)
                results_dict["valid_loss"].append(loss_t.item())
                results_dict["valid_precip_diffusion_loss"].append(loss_t.item())
                if self.rank == 0:
                    pbar.set_description("Epoch: {} valid_loss: {:.6f}".format(epoch, np.mean(results_dict["valid_loss"])))

        pbar.close()
        self.model.train()
        torch.cuda.empty_cache()
        gc.collect()
        for key in ["acc", "mae", "forecast_len"]:
            results_dict[f"valid_{key}"] = [0.0]
        return results_dict
