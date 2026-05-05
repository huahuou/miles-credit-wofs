"""
Training application for WoFS DA increment learning with ensemble KCRPS.

This is the ensemble companion to train_wrf_wofs_da.py. It keeps the same
DA dataset and Aurora-compatible model wiring, but enables ensemble members
inside the single-step WRF trainer and trains them with KCRPS.
"""

import logging
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import optuna
import torch
import warnings
import yaml
from echo.src.base_objective import BaseObjective

from train_wrf import load_model_states_and_optimizer
from train_wrf_wofs_da import (
    _configure_aurora_da_model,
    _resolve_loader_runtime,
    _select_files,
    _strip_deprecated_concentration_config,
    _sync_prognostic_levels,
    _validate_training_transform_config,
    _worker_init_fn,
)
from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.losses.weighted_loss import VariableTotalLoss2D
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.parser import credit_main_parser
from credit.pbs import launch_script, launch_script_mpi
from credit.scheduler import annealed_probability
from credit.seed import seed_everything
from credit.samplers import DistributedFileLocalitySampler
from credit.trainers import load_trainer
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_dataset_and_sampler(conf, param_interior, param_outside, world_size, rank, is_train=True):
    seed = conf["seed"]
    dataset = WoFSDAIncrementDataset(
        param_interior,
        param_outside,
        conf=conf,
        seed=seed,
    )

    use_file_locality_sampler = bool(conf["trainer"].get("use_file_locality_sampler", False)) and is_train
    if use_file_locality_sampler:
        sampler = DistributedFileLocalitySampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=is_train,
            drop_last=True,
        )

    logging.info("Loaded WoFS DA increment dataset and distributed sampler")
    return dataset, sampler


def _build_params(conf, split: str):
    years_range = conf["data"]["train_years"] if split == "train" else conf["data"]["valid_years"]
    date_range = conf["data"].get("train_date_range") if split == "train" else conf["data"].get("valid_date_range")
    history_key = "history_len" if split == "train" else "valid_history_len"
    forecast_key = "forecast_len" if split == "train" else "valid_forecast_len"

    filenames = _select_files(conf["data"]["save_loc"], years_range, date_range)
    filename_dyn_forcing = (
        _select_files(conf["data"]["save_loc_dynamic_forcing"], years_range, date_range)
        if conf["data"].get("save_loc_dynamic_forcing")
        else None
    )

    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_context_upper_air": conf["data"]["context_upper_air_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "filenames": filenames,
        "filename_dyn_forcing": filename_dyn_forcing,
        "history_len": conf["data"][history_key],
        "forecast_len": conf["data"][forecast_key],
    }

    param_outside = {
        "varname_upper_air": conf["data"]["observation_variables"],
    }

    return param_interior, param_outside


def main(rank, world_size, conf, backend, trial=False):
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    _sync_prognostic_levels(conf)
    _configure_aurora_da_model(conf)

    distributed = conf["trainer"]["mode"] in ["fsdp", "ddp"]
    try:
        if distributed:
            setup(rank, world_size, conf["trainer"]["mode"], backend)

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(rank % torch.cuda.device_count())
        else:
            device = torch.device("cpu")

        seed_everything(conf["seed"])

        train_batch_size = conf["trainer"]["train_batch_size"]
        valid_batch_size = conf["trainer"]["valid_batch_size"]
        thread_workers, train_prefetch_factor, train_pin_memory, train_persistent_workers, train_timeout_s = _resolve_loader_runtime(
            conf, "train"
        )
        valid_thread_workers, valid_prefetch_factor, valid_pin_memory, valid_persistent_workers, valid_timeout_s = _resolve_loader_runtime(
            conf, "valid"
        )

        if conf["data"]["scaler_type"] != "std-wrf":
            raise ValueError("WoFS DA increment app currently supports scaler_type='std-wrf' only")

        param_interior_train, param_outside_train = _build_params(conf, "train")
        param_interior_valid, param_outside_valid = _build_params(conf, "valid")

        train_dataset, train_sampler = load_dataset_and_sampler(
            conf, param_interior_train, param_outside_train, world_size, rank, is_train=True
        )
        valid_dataset, valid_sampler = load_dataset_and_sampler(
            conf, param_interior_valid, param_outside_valid, world_size, rank, is_train=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=train_pin_memory,
            persistent_workers=train_persistent_workers,
            num_workers=thread_workers,
            drop_last=True,
            prefetch_factor=train_prefetch_factor if thread_workers > 0 else None,
            worker_init_fn=_worker_init_fn,
            timeout=train_timeout_s,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            sampler=valid_sampler,
            pin_memory=valid_pin_memory,
            persistent_workers=valid_persistent_workers,
            num_workers=valid_thread_workers,
            drop_last=True,
            prefetch_factor=valid_prefetch_factor if valid_thread_workers > 0 else None,
            worker_init_fn=_worker_init_fn,
            timeout=valid_timeout_s,
        )

        m = load_model(conf)
        m.to(device)

        if conf["trainer"].get("compile", False):
            m = torch.compile(m)

        if distributed:
            model = distributed_model_wrapper(conf, m, device)
        else:
            model = m

        conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

        train_criterion = VariableTotalLoss2D(conf)
        valid_criterion = VariableTotalLoss2D(conf, validation=True)
        metrics = LatWeightedMetrics(conf)

        trainer_cls = load_trainer(conf)
        trainer = trainer_cls(model, rank)

        return trainer.fit(
            conf,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            train_criterion=train_criterion,
            valid_criterion=valid_criterion,
            scaler=scaler,
            scheduler=scheduler,
            metrics=metrics,
            rollout_scheduler=annealed_probability,
            trial=trial,
        )
    finally:
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        conf["model"]["dim_head"] = conf["model"]["dim"]
        conf["model"]["vq_codebook_dim"] = conf["model"]["dim"]
        conf.setdefault("loss", {})
        conf["loss"]["training_loss"] = "KCRPS"
        conf["loss"]["validation_loss"] = conf["loss"].get("validation_loss", "KCRPS")
        conf.setdefault("trainer", {})
        conf["trainer"]["type"] = "standard-wrf"
        conf["trainer"].setdefault("distributed_ensemble", False)
        conf["trainer"].setdefault("ensemble_size", 8)
        conf.setdefault("model", {}).setdefault("noise_injection", {})["activate"] = True
        try:
            return main(0, 1, conf, backend="nccl", trial=trial)
        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                logging.warning(f"Pruning trial {trial.number} due to runtime error: {str(E)}.")
                raise optuna.TrialPruned()
            raise E


def primary_main():
    parser = ArgumentParser(description="Train a WoFS DA increment ensemble model with KCRPS")
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("-l", dest="launch", type=int, default=0, help="Submit workers to PBS.")
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distributed training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf.setdefault("trainer", {})
    conf["trainer"]["type"] = "standard-wrf"
    conf["trainer"].setdefault("ensemble_size", 8)
    ensemble_size = int(conf["trainer"].get("ensemble_size", 1))
    conf.setdefault("loss", {})
    conf["loss"]["training_loss"] = "KCRPS"
    conf["loss"].setdefault("validation_loss", "KCRPS")

    noise_conf = conf.setdefault("model", {}).setdefault("noise_injection", {})
    noise_conf["activate"] = True
    noise_conf.setdefault("latent_dim", 128)
    noise_conf.setdefault("noise_scales", [0.30, 0.15, 0.05, 0.01])
    noise_conf.setdefault("encoder_noise", True)
    noise_conf.setdefault("correlated", False)
    noise_conf.setdefault("sample_latent_if_none", True)
    noise_conf.setdefault("freeze_base_model_weights", False)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    _strip_deprecated_concentration_config(conf)
    _validate_training_transform_config(conf)
    prog_levels = _sync_prognostic_levels(conf)
    _configure_aurora_da_model(conf)
    logging.info("Using prognostic vertical levels: %s", prog_levels)

    if ensemble_size <= 1:
        logging.warning("trainer.ensemble_size <= 1; KCRPS path still works, but this is not a true ensemble run.")
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(args.model_config, os.path.join(save_loc, "model.yml"))

    if int(args.launch):
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(args.model_config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(args.model_config, script_path)
        sys.exit()

    seed_everything(conf["seed"])
    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    main(world_rank, world_size, conf, args.backend)


if __name__ == "__main__":
    primary_main()
