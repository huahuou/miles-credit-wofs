"""
TrainerMAE — Multi-modal Masked Autoencoder Trainer for WoFS DA
----------------------------------------------------------------
Registered as ``standard-wrf-mae`` in the trainer registry.

Interface changes versus TrainerWRF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Batch is a plain dict  {modality_key: (B, C, H, W)} rather than the
  x/y/x_boundary/x_surf/... layout.  No concat_and_reshape needed.
* The model is responsible for masking and reconstruction; the trainer
  only computes the weighted masked-MSE and passes gradients.
* Loss = Σ_m  λ_m  ·  mean over masked tokens of  ||x̂_m - x_m||²
  Default weights: background=0.5, precip=2.0, reflectivity=1.0.
* validate(): same modality dict feed, mask_inputs=True for proxy
  training-style validation OR mask_inputs=False for full-field RMSE.
  Controlled by ``conf["trainer"]["val_mask_inputs"]`` (default True).
"""

import gc
import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, IterableDataset

from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

# Default per-modality MSE loss weights
DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "background": 0.5,
    "precip": 2.0,
    "reflectivity": 1.0,
}


def _accum_log(log: dict, new_logs: dict):
    for key, value in new_logs.items():
        if key not in log:
            log[key] = value
        else:
            log[key] += value


class TrainerMAE(BaseTrainer):
    """
    Trainer for WoFSMultiModalMAE.

    Expected conf layout::

        trainer:
          type: standard-wrf-mae
          batches_per_epoch: 200
          valid_batches_per_epoch: 50
          grad_accum_every: 1
          grad_max_norm: 1.0
          amp: true
          loss_weights:                    # optional overrides
            background: 0.5
            precip: 2.0
            reflectivity: 1.0
          val_mask_inputs: true            # use masked forward during validation
          num_encoded_tokens: 256          # override model default during training
          dirichlet_alpha: 1.0
    """

    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)

    # ------------------------------------------------------------------ #
    # Masked MSE loss
    # ------------------------------------------------------------------ #
    def _mae_loss(
        self,
        recon_dict: Dict[str, torch.Tensor],
        orig_dict: Dict[str, torch.Tensor],
        task_masks: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-modality masked MSE and weighted total.

        Args:
            recon_dict  : {mod: (B, C, H, W)} — reconstructed fields
            orig_dict   : {mod: (B, C, H, W)} — original normalized fields
            task_masks  : {mod: (B, N_tok)} — 1=masked, 0=visible
            loss_weights: {mod: float}

        Returns:
            dict with 'total' and per-modality losses.
        """
        patch_size = getattr(self.model, "patch_size", 8)
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=self.device)

        for mod, recon in recon_dict.items():
            if mod not in orig_dict:
                continue
            orig = orig_dict[mod].to(self.device, non_blocking=True)
            err = (recon - orig) ** 2  # (B, C, H, W)

            mask_tokens = task_masks.get(mod)  # (B, N_tok)  1=masked
            if mask_tokens is not None and mask_tokens.any():
                B, C, H, W = err.shape
                N_H = H // patch_size
                N_W = W // patch_size
                # Reshape err to patch space for token-level masking
                # (B, C, N_H, P, N_W, P) → (B, N_tok, C*P*P)
                err_patches = err[:, :, :N_H * patch_size, :N_W * patch_size]
                err_patches = err_patches.reshape(B, C, N_H, patch_size, N_W, patch_size)
                err_patches = err_patches.permute(0, 2, 4, 1, 3, 5).reshape(B, N_H * N_W, -1)
                # Align mask length with patch count
                m = mask_tokens.float()  # (B, N_tok)
                n_tok = min(err_patches.shape[1], m.shape[1])
                err_patches = err_patches[:, :n_tok, :]
                m = m[:, :n_tok]  # (B, N_tok)
                # Mean per-element MSE within each patch, then mean over masked patches.
                # err_patches: (B, N_tok, C*P*P) → err_per_patch: (B, N_tok)
                err_per_patch = err_patches.mean(-1)
                denom = m.sum().clamp_min(1.0)
                mod_loss = (err_per_patch * m).sum() / denom
            else:
                # If no mask info, average over all pixels
                mod_loss = err.mean()

            w = loss_weights.get(mod, 1.0)
            losses[mod] = mod_loss
            total = total + w * mod_loss

        losses["total"] = total
        return losses

    # ------------------------------------------------------------------ #
    # train_one_epoch
    # ------------------------------------------------------------------ #
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        trainer_conf = conf["trainer"]
        batches_per_epoch = trainer_conf["batches_per_epoch"]
        grad_accum_every = trainer_conf.get("grad_accum_every", 1)
        grad_max_norm = trainer_conf.get("grad_max_norm", None)
        amp = trainer_conf.get("amp", True)
        distributed = trainer_conf.get("mode", "none") in ("fsdp", "ddp")

        loss_weights = {**DEFAULT_LOSS_WEIGHTS, **trainer_conf.get("loss_weights", {})}
        num_encoded_tokens = trainer_conf.get("num_encoded_tokens", None)
        dirichlet_alpha = float(trainer_conf.get("dirichlet_alpha", 1.0))

        # Update LR scheduler for epoch-level schedulers
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        if not isinstance(trainloader.dataset, IterableDataset):
            batches_per_epoch = (
                batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)
            )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=(self.rank > 0),
        )

        results_dict = defaultdict(list)

        # Use a manual restart iterator instead of itertools.cycle.
        # itertools.cycle saves every yielded element internally for replay,
        # which would accumulate hundreds of GB of tensor data when batches_per_epoch
        # is large (e.g., the full DataLoader length of 7000+ batches × 200 MB/batch).
        dl_iter = iter(trainloader)

        optimizer.zero_grad()

        for i in batch_group_generator:
            try:
                batch = next(dl_iter)
            except StopIteration:
                # DataLoader exhausted — restart without accumulating old batches
                dl_iter = iter(trainloader)
                batch = next(dl_iter)
            logs: dict = {}

            # Move batch modalities to device
            modality_dict = {
                k: v.to(self.device, non_blocking=True)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "index"
            }

            with autocast(enabled=amp):
                recon_dict, task_masks = self.model(
                    modality_dict,
                    mask_inputs=True,
                    num_encoded_tokens=num_encoded_tokens,
                    alphas=dirichlet_alpha,
                )

                losses = self._mae_loss(recon_dict, modality_dict, task_masks, loss_weights)
                loss = losses["total"]

            # Backward
            scaler.scale(loss / grad_accum_every).backward()
            _accum_log(logs, {"loss": loss.item() / grad_accum_every})

            if (i + 1) % grad_accum_every == 0:
                if grad_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Reduce and log
            batch_loss = torch.tensor([logs["loss"]], device=self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())

            # Per-modality losses
            for mod, mod_loss in losses.items():
                if mod == "total":
                    continue
                ml_t = torch.tensor([mod_loss.item()], device=self.device)
                if distributed:
                    dist.all_reduce(ml_t, dist.ReduceOp.AVG, async_op=False)
                results_dict[f"train_{mod}_loss"].append(ml_t[0].item())

            # tqdm description
            to_print = "Epoch: {} train_loss: {:.6f}".format(
                epoch, np.mean(results_dict["train_loss"])
            )
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                logger.warning("Non-finite loss encountered, stopping epoch early.")
                break

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()

        # Populate required base_trainer keys
        for key in ["acc", "mae", "forecast_len"]:
            if f"train_{key}" not in results_dict:
                results_dict[f"train_{key}"] = [0.0]

        return results_dict

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        trainer_conf = conf["trainer"]
        valid_batches = trainer_conf.get("valid_batches_per_epoch", 0)
        amp = trainer_conf.get("amp", True)
        distributed = trainer_conf.get("mode", "none") in ("fsdp", "ddp")
        val_mask_inputs = trainer_conf.get("val_mask_inputs", True)
        loss_weights = {**DEFAULT_LOSS_WEIGHTS, **trainer_conf.get("loss_weights", {})}
        num_encoded_tokens = trainer_conf.get("num_encoded_tokens", None)
        dirichlet_alpha = float(trainer_conf.get("dirichlet_alpha", 1.0))

        n_valid = valid_batches if 0 < valid_batches < len(valid_loader) else len(valid_loader)

        batch_group_generator = tqdm.tqdm(
            range(n_valid),
            total=n_valid,
            leave=True,
            disable=(self.rank > 0),
        )

        results_dict = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            dl_iter = iter(valid_loader)
            for i in batch_group_generator:
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    break

                modality_dict = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k != "index"
                }

                with autocast(enabled=amp):
                    recon_dict, task_masks = self.model(
                        modality_dict,
                        mask_inputs=val_mask_inputs,
                        num_encoded_tokens=num_encoded_tokens,
                        alphas=dirichlet_alpha,
                    )

                    losses = self._mae_loss(recon_dict, modality_dict, task_masks, loss_weights)
                    loss = losses["total"]

                batch_loss = torch.tensor([loss.item()], device=self.device)
                if distributed:
                    torch.distributed.barrier()
                    dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                results_dict["valid_loss"].append(batch_loss[0].item())

                for mod, mod_loss in losses.items():
                    if mod == "total":
                        continue
                    ml_t = torch.tensor([mod_loss.item()], device=self.device)
                    if distributed:
                        dist.all_reduce(ml_t, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"valid_{mod}_loss"].append(ml_t[0].item())

                to_print = "Epoch: {} valid_loss: {:.6f}".format(
                    epoch, np.mean(results_dict["valid_loss"])
                )
                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

        batch_group_generator.close()

        self.model.train()
        torch.cuda.empty_cache()
        gc.collect()

        # Populate required base_trainer keys
        for key in ["acc", "mae", "forecast_len"]:
            if f"valid_{key}" not in results_dict:
                results_dict[f"valid_{key}"] = [0.0]

        return results_dict
