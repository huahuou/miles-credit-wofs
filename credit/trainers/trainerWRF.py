""" """

import gc
import tqdm
import logging
from collections import defaultdict

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

import optuna
from credit.data import concat_and_reshape, reshape_only
from credit.losses.weighted_loss import binary_focal_with_logits, soft_dice_loss_from_logits
from credit.scheduler import update_on_batch
from credit.trainers.utils import accum_log, cycle
from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)

        logger.info("WRF single-step training")

    def _occupancy_channel_mask(self, conf, num_prog, dtype):
        occ_conf = conf.get("loss", {}).get("occupancy", {})
        configured_vars = occ_conf.get("variables", conf["data"]["variables"])
        if configured_vars is None:
            configured_vars = []
        occ_vars = set(configured_vars)

        prog_vars = conf["data"]["variables"]
        n_vars = len(prog_vars)
        if n_vars <= 0 or num_prog <= 0:
            return torch.zeros((1, num_prog, 1, 1, 1), device=self.device, dtype=dtype)

        if num_prog % n_vars != 0:
            # Fallback: apply occupancy to all prognostic channels if layout is unexpected.
            return torch.ones((1, num_prog, 1, 1, 1), device=self.device, dtype=dtype)

        levels = num_prog // n_vars
        channel_mask = torch.zeros((num_prog,), device=self.device, dtype=dtype)
        for var_idx, var_name in enumerate(prog_vars):
            if var_name in occ_vars:
                start = var_idx * levels
                end = start + levels
                channel_mask[start:end] = 1.0

        return channel_mask.view(1, num_prog, 1, 1, 1)

    def _compute_loss_with_occupancy(self, conf, batch, y, y_pred, criterion, varnum_diag):
        occ_conf = conf.get("loss", {}).get("occupancy", {})
        occ_enabled = bool(occ_conf.get("enabled", False))

        reg_loss = criterion(y, y_pred)
        occ_ce = None
        gate_mean = None

        if not occ_enabled:
            return reg_loss, reg_loss, occ_ce, gate_mean

        required_keys = ("y_occupancy", "y_occ_delta_threshold_norm")
        if not all(key in batch for key in required_keys):
            return reg_loss, reg_loss, occ_ce, gate_mean

        num_prog = y_pred.shape[1] - varnum_diag
        if num_prog <= 0:
            return reg_loss, reg_loss, occ_ce, gate_mean

        tau = float(occ_conf.get("logit_temperature", 0.25))
        tau = max(tau, 1.0e-6)
        ce_weight = float(occ_conf.get("ce_weight", 0.2))
        use_masked_reg = bool(occ_conf.get("use_masked_regression", True))
        masked_mode = str(occ_conf.get("masked_regression_mode", "soft")).lower().strip()
        reg_mask_min = float(occ_conf.get("reg_mask_min", 0.05))

        y_prog_pred = y_pred[:, :num_prog, ...]
        y_occ = reshape_only(batch["y_occupancy"]).to(self.device, dtype=y_pred.dtype, non_blocking=True)
        delta_thr_norm = reshape_only(batch["y_occ_delta_threshold_norm"]).to(
            self.device, dtype=y_pred.dtype, non_blocking=True
        )

        occ_logits = (torch.abs(y_prog_pred) - delta_thr_norm) / tau
        occ_channel_mask = self._occupancy_channel_mask(conf, num_prog, y_pred.dtype)
        occ_weight = occ_channel_mask.expand_as(occ_logits)

        pos_weight = occ_conf.get("pos_weight", None)
        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(float(pos_weight), device=self.device, dtype=y_pred.dtype)

        occ_loss_map = F.binary_cross_entropy_with_logits(
            occ_logits,
            y_occ,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        occ_denom = occ_weight.sum().clamp_min(1.0)
        occ_ce = (occ_loss_map * occ_weight).sum() / occ_denom

        occ_aux_loss = torch.zeros((), device=self.device, dtype=y_pred.dtype)
        if bool(occ_conf.get("use_focal", False)):
            occ_aux_loss = occ_aux_loss + float(occ_conf.get("focal_weight", 0.0)) * binary_focal_with_logits(
                occ_logits,
                y_occ,
                alpha=float(occ_conf.get("focal_alpha", 0.25)),
                gamma=float(occ_conf.get("focal_gamma", 2.0)),
                mask=occ_weight,
            )
        if bool(occ_conf.get("use_dice", False)):
            occ_aux_loss = occ_aux_loss + float(occ_conf.get("dice_weight", 0.0)) * soft_dice_loss_from_logits(
                occ_logits,
                y_occ,
                smooth=float(occ_conf.get("dice_smooth", 1.0)),
                mask=occ_weight,
            )

        if use_masked_reg:
            if masked_mode == "hard" and "y_regression_mask_hard" in batch:
                reg_mask_prog = reshape_only(batch["y_regression_mask_hard"]).to(
                    self.device, dtype=y_pred.dtype, non_blocking=True
                )
            else:
                reg_floor = torch.tensor(reg_mask_min, device=self.device, dtype=y_pred.dtype)
                reg_mask_prog = torch.maximum(torch.sigmoid(occ_logits).detach(), reg_floor)

            # Channels not selected for occupancy keep full regression supervision.
            reg_mask_prog = reg_mask_prog * occ_channel_mask + (1.0 - occ_channel_mask)

            full_reg_mask = torch.ones_like(y_pred)
            full_reg_mask[:, :num_prog, ...] = reg_mask_prog
            reg_loss = criterion(y, y_pred, mask=full_reg_mask)
        else:
            reg_loss = criterion(y, y_pred)

        gate_mean = (torch.sigmoid(occ_logits).detach() * occ_weight).sum() / occ_denom
        total_loss = reg_loss + ce_weight * (occ_ce + occ_aux_loss)
        # print(reg_loss.item(), occ_ce.item(), occ_aux_loss.item(), gate_mean.item())
        return total_loss, reg_loss, occ_ce, gate_mean

    # Training function.
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        # training hyperparameters
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        grad_max_norm = conf["trainer"]["grad_max_norm"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        residual_prediction = conf["trainer"].get("residual_prediction", False)
        retain_graph = conf["data"].get("retain_graph", False)
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        # forecast step
        if "total_time_steps" in conf["data"]:
            total_time_steps = conf["data"]["total_time_steps"]
        else:
            total_time_steps = forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        # ====================================================== #

        # set up a custom tqdm
        if not isinstance(trainloader.dataset, IterableDataset):
            # if batches_per_epoch = 0, use all training samples (i.e., full epoch)
            batches_per_epoch = batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        results_dict = defaultdict(list)

        # dataloader
        dl = cycle(trainloader)

        for i in batch_group_generator:
            # Get the next batch from the iterator
            batch = next(dl)

            # training log
            logs = {}

            with autocast(enabled=amp):
                # --------------------------------------------------------------------------------- #
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device, non_blocking=True)
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(self.device, non_blocking=True)

                # --------------------------------------------------------------------------------- #
                # add forcing and static variables
                if "x_forcing_static" in batch:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    x_forcing_batch = batch["x_forcing_static"].to(self.device, non_blocking=True).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # combine y and y_surf
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device, non_blocking=True)
                else:
                    y = reshape_only(batch["y"]).to(self.device, non_blocking=True)

                if "y_diag" in batch:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    y_diag_batch = batch["y_diag"].to(self.device, non_blocking=True).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    y = torch.cat((y, y_diag_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # boundary conditions
                if "x_surf_boundary" in batch:
                    x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(self.device, non_blocking=True)
                else:
                    x_boundary = reshape_only(batch["x_boundary"]).to(self.device, non_blocking=True)

                # --------------------------------------------------------------------------------- #
                # time encoding
                x_time_encode = batch["x_time_encode"].to(self.device, non_blocking=True)

                # single step predict
                y_pred = self.model(x, x_boundary, x_time_encode)

                # Residual prediction: model outputs a delta, add last input state back
                if residual_prediction:
                    num_prog = y_pred.shape[1] - varnum_diag
                    residual = x[:, :num_prog, -1:, ...]
                    if varnum_diag > 0:
                        y_pred = torch.cat(
                            [y_pred[:, :num_prog, ...] + residual,
                             y_pred[:, num_prog:, ...]],
                            dim=1,
                        )
                    else:
                        y_pred = y_pred + residual

                y = y.to(device=self.device, dtype=y_pred.dtype, non_blocking=True)

                # loss compute
                loss, reg_loss, occ_ce, gate_mean = self._compute_loss_with_occupancy(
                    conf,
                    batch,
                    y,
                    y_pred,
                    criterion,
                    varnum_diag,
                )

                # Metrics
                # metrics_dict = metrics(y_pred.float(), y.float())
                metrics_dict = metrics(y_pred, y)

                # save training metrics
                for name, value in metrics_dict.items():
                    value = torch.tensor([value], device=self.device)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                # backpropagation
                loss = loss.mean()

                scaler.scale(loss / grad_accum_every).backward(retain_graph=retain_graph)

            accum_log(logs, {"loss": loss.item() / grad_accum_every})
            if reg_loss is not None:
                accum_log(logs, {"reg_loss": reg_loss.item() / grad_accum_every})
            if occ_ce is not None:
                accum_log(logs, {"occ_ce": occ_ce.item() / grad_accum_every})
            if gate_mean is not None:
                accum_log(logs, {"occ_gate_mean": gate_mean.item()})


            # Note: DDP synchronizes gradients during backward(); no explicit barrier needed.

            if grad_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)

            scaler.step(optimizer)
            scaler.update()

            # clear grad
            optimizer.zero_grad()

            # Handle batch_loss
            batch_loss = torch.tensor([logs["loss"]], device=self.device)

            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)

            results_dict["train_loss"].append(batch_loss[0].item())

            if "reg_loss" in logs:
                reg_loss_tensor = torch.tensor([logs["reg_loss"]], device=self.device)
                if distributed:
                    dist.all_reduce(reg_loss_tensor, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_reg_loss"].append(reg_loss_tensor[0].item())

            if "occ_ce" in logs:
                occ_ce_tensor = torch.tensor([logs["occ_ce"]], device=self.device)
                if distributed:
                    dist.all_reduce(occ_ce_tensor, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_occ_ce"].append(occ_ce_tensor[0].item())

            if "occ_gate_mean" in logs:
                occ_gate_tensor = torch.tensor([logs["occ_gate_mean"]], device=self.device)
                if distributed:
                    dist.all_reduce(occ_gate_tensor, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_occ_gate_mean"].append(occ_gate_tensor[0].item())

            if "forecast_hour" in batch:
                forecast_hour_tensor = batch["forecast_hour"].to(self.device, non_blocking=True)
                if distributed:
                    dist.all_reduce(forecast_hour_tensor, dist.ReduceOp.AVG, async_op=False)
                    forecast_hour_avg = forecast_hour_tensor[-1].item()
                else:
                    forecast_hour_avg = batch["forecast_hour"][-1].item()

                results_dict["train_forecast_len"].append(forecast_hour_avg + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    print("Invalid loss value: {}".format(np.mean(results_dict["train_loss"])))
                    raise optuna.TrialPruned()

                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                np.mean(results_dict["train_forecast_len"]),
            )

            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch:
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]

        forecast_len = conf["data"]["valid_forecast_len"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False
        residual_prediction = conf["trainer"].get("residual_prediction", False)
        varnum_diag = len(conf["data"]["diagnostic_variables"])

        total_time_steps = conf["data"]["total_time_steps"] if "total_time_steps" in conf["data"] else forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        results_dict = defaultdict(list)

        # ====================================================== #

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = (
                valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)
            )

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        dl = cycle(valid_loader)

        for i in batch_group_generator:
            batch = next(dl)

            with torch.no_grad():
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device, non_blocking=True)
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(self.device, non_blocking=True)

                # --------------------------------------------------------------------------------- #
                # add forcing and static variables
                if "x_forcing_static" in batch:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    x_forcing_batch = batch["x_forcing_static"].to(self.device, non_blocking=True).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # combine y and y_surf
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device, non_blocking=True)
                else:
                    y = reshape_only(batch["y"]).to(self.device, non_blocking=True)

                if "y_diag" in batch:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    y_diag_batch = batch["y_diag"].to(self.device, non_blocking=True).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    y = torch.cat((y, y_diag_batch), dim=1)

                # --------------------------------------------------------------------------------- #
                # boundary conditions
                if "x_surf_boundary" in batch:
                    x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(self.device, non_blocking=True)
                else:
                    x_boundary = reshape_only(batch["x_boundary"]).to(self.device, non_blocking=True)

                # --------------------------------------------------------------------------------- #
                # time encoding
                x_time_encode = batch["x_time_encode"].to(self.device, non_blocking=True)

                y_pred = self.model(x, x_boundary, x_time_encode)

                # Residual prediction: model outputs a delta, add last input state back
                if residual_prediction:
                    num_prog = y_pred.shape[1] - varnum_diag
                    residual = x[:, :num_prog, -1:, ...]
                    if varnum_diag > 0:
                        y_pred = torch.cat(
                            [y_pred[:, :num_prog, ...] + residual,
                             y_pred[:, num_prog:, ...]],
                            dim=1,
                        )
                    else:
                        y_pred = y_pred + residual

                y = y.to(device=self.device, dtype=y_pred.dtype, non_blocking=True)
                loss, reg_loss, occ_ce, gate_mean = self._compute_loss_with_occupancy(
                    conf,
                    batch,
                    y,
                    y_pred,
                    criterion,
                    varnum_diag,
                )

                # Metrics
                # metrics_dict = metrics(y_pred, y)
                metrics_dict = metrics(y_pred.float(), y.float())

                for name, value in metrics_dict.items():
                    value = torch.tensor([value], device=self.device)

                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)

                    results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.tensor([loss.item()], device=self.device)

                if distributed:
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                if reg_loss is not None:
                    results_dict["valid_reg_loss"].append(reg_loss.item())
                if occ_ce is not None:
                    results_dict["valid_occ_ce"].append(occ_ce.item())
                if gate_mean is not None:
                    results_dict["valid_occ_gate_mean"].append(gate_mean.item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )

                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

                if i >= valid_batches_per_epoch and i > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
