"""Trainer for WoFS conditional DiffMAE precip inpainting."""

from __future__ import annotations

import gc
import logging
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
            float(conf.get("mixed_height_channel_probability", 1.0)),
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

        if mode in {"mixed", "spatial_or_channel"} and not validation:
            channel_prob = float(trainer_conf.get("channel_patch_mask_probability", 0.5))
            mode = "channel_patch" if torch.rand((), device=device).item() < channel_prob else "spatial_patch"
        elif mode in {"mixed_height", "mixed_with_height", "spatial_channel_height"}:
            mode = _choose_mixed_height_mode(trainer_conf, device)

        if mode in {"height_patch", "height", "level_patch", "vertical_patch"}:
            return model.random_height_precip_mask(
                batch_size,
                ratio,
                device,
                masked_levels=trainer_conf.get("height_mask_levels"),
                visible_levels=trainer_conf.get("height_visible_levels"),
            )
        if mode in {"channel_patch", "random_channel", "channel"}:
            return model.random_channel_precip_mask(batch_size, ratio, device)
        if mode in {"spatial_patch", "spatial", "patch"}:
            return model.random_precip_mask(batch_size, ratio, device)
        raise ValueError(f"Unsupported precip mask mode: {mode!r}")

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
        model = self._unwrap_model()
        mask_img_full = model.expand_patch_mask(precip_mask[:1], target.shape[-2], target.shape[-1])
        mask_img = mask_img_full.amax(dim=1, keepdim=True)
        target_1 = target[0]
        noisy_1 = noisy[0]
        pred_1 = pred_x0[0]
        masked_abs_err = (pred_1 - target_1).abs() * mask_img_full[0]

        summary_target = target_1.mean(dim=0)
        summary_noisy = noisy_1.mean(dim=0)
        summary_pred = pred_1.mean(dim=0)
        summary_err = masked_abs_err.mean(dim=0)

        ncols = 6 if sample is not None else 5
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
        ch_mask = mask_img_full[0, ch if mask_img_full.shape[1] > 1 else 0]
        self._plot_field(axes[1, 3], (pred_1[ch] - target_1[ch]).abs() * ch_mask, f"masked err ch{ch}", cmap="magma")
        self._plot_field(axes[1, 4], ch_mask, f"mask ch{ch}", cmap="gray")

        if sample is not None:
            sample_1 = sample[0]
            self._plot_field(axes[0, 5], sample_1.mean(dim=0), "sample mean")
            self._plot_field(axes[1, 5], sample_1[ch], f"sample ch{ch}")

        fig.savefig(out_path, dpi=140)
        plt.close(fig)

    def _maybe_save_denoise_snapshot(self, epoch: int, step: int, conf: dict, batch: dict, losses: dict) -> None:
        snapshot_conf = conf["trainer"].get("denoise_snapshot", {})
        if not snapshot_conf or not snapshot_conf.get("enabled", False) or self.rank != 0:
            return
        every = int(snapshot_conf.get("every_steps", 0))
        every_epoch = bool(snapshot_conf.get("every_epoch", False))
        should_save_step = every > 0 and step % every == 0
        should_save_epoch = every_epoch and step == 0
        if not (should_save_step or should_save_epoch):
            return
        out_dir = Path(conf["save_loc"]) / snapshot_conf.get("dir", "denoise_snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        model = self._unwrap_model()
        with torch.no_grad():
            pred_noise, pred_x0 = model.model_predictions(
                losses["x_t"][:1],
                losses["t"][:1],
                self._condition_dict({k: v[:1] for k, v in batch.items()}),
                losses["precip_mask"][:1],
            )
            sample = None
            if bool(snapshot_conf.get("save_sampling_figure", False)):
                sampling_steps = int(snapshot_conf.get("sampling_timesteps", 8))
                sample_visible = batch["precip"][:1] if bool(snapshot_conf.get("sampling_use_visible_precip", True)) else None
                sample = model.sample_precip(
                    self._condition_dict({k: v[:1] for k, v in batch.items()}),
                    losses["precip_mask"][:1],
                    precip_visible=sample_visible,
                    sampling_timesteps=sampling_steps,
                )
            if bool(snapshot_conf.get("save_snapshot", True)):
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "x_t": losses["x_t"][:1].detach().cpu(),
                        "target_precip": batch["precip"][:1].detach().cpu(),
                        "pred_x0": pred_x0[:1].detach().cpu(),
                        "pred_noise": pred_noise[:1].detach().cpu(),
                        "precip_mask": losses["precip_mask"][:1].detach().cpu(),
                        "sample": None if sample is None else sample[:1].detach().cpu(),
                    },
                    out_dir / f"epoch{epoch:04d}_step{step:06d}.pt",
                )
            if bool(snapshot_conf.get("save_figure", True)):
                self._save_denoise_figure(
                    out_dir / f"epoch{epoch:04d}_step{step:06d}.png",
                    target=batch["precip"][:1],
                    noisy=losses["x_t"][:1],
                    pred_x0=pred_x0[:1],
                    precip_mask=losses["precip_mask"][:1],
                    sample=None if sample is None else sample[:1],
                    channel=int(snapshot_conf.get("channel", 0)),
                )

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
            precip_mask = self._sample_mask(batch["precip"].shape[0], trainer_conf, batch["precip"].device)

            with autocast(enabled=amp):
                losses = model.p_losses(batch["precip"], self._condition_dict(batch), precip_mask)
                loss = losses["loss"]

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
                logger.warning("Non-finite DiffMAE loss encountered, stopping epoch early.")
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
                precip_mask = self._sample_mask(batch["precip"].shape[0], trainer_conf, batch["precip"].device, validation=True)
                with autocast(enabled=amp):
                    losses = model.p_losses(batch["precip"], self._condition_dict(batch), precip_mask)
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
