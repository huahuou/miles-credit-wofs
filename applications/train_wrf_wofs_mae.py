"""
Training application for WoFS Multi-Modal Masked Autoencoder (MAE).

Single-stage self-supervised pre-training; no VAE required.
Trains WoFSMultiModalMAE on WoFS NWP zarr output.

Usage (single-node, 2 GPUs):
    torchrun --standalone --nnodes=1 --nproc-per-node=2 \\
        applications/train_wrf_wofs_mae.py \\
        -c config/wofs_mae.yml

Usage (PBS/SLURM with mpirun):
    mpirun -n 4 python applications/train_wrf_wofs_mae.py -c config/wofs_mae.yml
"""

import logging
import os
import re
import shutil
import sys
import warnings
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import torch
import yaml
from torch.utils.data.distributed import DistributedSampler

from train_wrf import load_model_states_and_optimizer
from credit.datasets.wrf_wofs_mae import WoFSMAEDataset
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.pbs import launch_script, launch_script_mpi
from credit.samplers import DistributedFileLocalitySampler
from credit.scheduler import annealed_probability
from credit.seed import seed_everything
from credit.trainers import load_trainer

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _worker_init_fn(worker_id):
    """Open zarr/xarray resources in each DataLoader worker."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    ds = worker_info.dataset
    if hasattr(ds, "_open_datasets") and callable(ds._open_datasets):
        try:
            ds._open_datasets()
        except Exception as exc:
            logging.warning("worker_init_fn failed in worker %d: %s", worker_id, exc)
            raise


def _extract_case_date(file_path: str):
    """Return YYYYMMDD string from a WoFS zarr filename, or None."""
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
    return match.group(1) if match else None


def _resolve_loader_runtime(conf: dict, training_type: str):
    trainer_conf = conf["trainer"]
    is_train = training_type == "train"
    workers = int(
        trainer_conf.get("thread_workers", 0)
        if is_train
        else trainer_conf.get("valid_thread_workers", trainer_conf.get("thread_workers", 0))
    )
    prefetch = int(
        trainer_conf.get("prefetch_factor", 4)
        if is_train
        else trainer_conf.get("valid_prefetch_factor", trainer_conf.get("prefetch_factor", 4))
    )
    pin_memory = bool(trainer_conf.get("pin_memory", True))
    persistent_workers = bool(trainer_conf.get("persistent_workers", workers > 0))
    timeout_s = int(trainer_conf.get("dataloader_timeout_s", 0))

    if bool(trainer_conf.get("no_hang_debug", False)):
        workers = int(
            trainer_conf.get("no_hang_thread_workers", 1)
            if is_train
            else trainer_conf.get("no_hang_valid_thread_workers", 1)
        )
        prefetch = int(
            trainer_conf.get("no_hang_prefetch_factor", 1)
            if is_train
            else trainer_conf.get("no_hang_valid_prefetch_factor", 1)
        )
        pin_memory = bool(trainer_conf.get("no_hang_pin_memory", False))
        persistent_workers = bool(trainer_conf.get("no_hang_persistent_workers", workers > 0))
        timeout_s = int(trainer_conf.get("no_hang_timeout_s", 180))

    if workers <= 0:
        prefetch = 0
        persistent_workers = False

    return workers, prefetch, pin_memory, persistent_workers, timeout_s


def _select_files(glob_pattern: str, years_range, date_range=None):
    files = sorted(glob(glob_pattern))

    if date_range is not None:
        start_date, end_date = str(date_range[0]), str(date_range[1])
        selected = []
        for fp in files:
            case_date = _extract_case_date(fp)
            if case_date is not None and start_date <= case_date <= end_date:
                selected.append(fp)
        return selected

    if years_range is None:
        return files

    if len(years_range) == 2:
        start_year, end_year = int(years_range[0]), int(years_range[1])
        years = [str(y) for y in range(start_year, end_year + 1)]
    else:
        years = [str(y) for y in years_range]

    return [fp for fp in files if any(yr in fp for yr in years)]


def load_dataset_and_sampler(conf: dict, filenames, world_size: int, rank: int, is_train: bool = True):
    """Build WoFSMAEDataset and its DistributedSampler."""
    seed = conf["seed"]
    dataset = WoFSMAEDataset(filenames=filenames, conf=conf, seed=seed)

    use_file_locality = bool(conf["trainer"].get("use_file_locality_sampler", False)) and is_train
    if use_file_locality:
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

    logging.info("Loaded WoFS MAE dataset with %d files (is_train=%s)", len(filenames), is_train)
    return dataset, sampler


def _build_filenames(conf: dict, split: str):
    years_range = conf["data"]["train_years"] if split == "train" else conf["data"]["valid_years"]
    date_range = (
        conf["data"].get("train_date_range")
        if split == "train"
        else conf["data"].get("valid_date_range")
    )
    filenames = _select_files(conf["data"]["save_loc"], years_range, date_range)
    if not filenames:
        raise ValueError(
            f"No {'training' if split == 'train' else 'validation'} files found "
            f"for pattern '{conf['data']['save_loc']}' with filters: "
            f"years={years_range}, date_range={date_range}"
        )
    return filenames


class _DummyMetrics:
    """Placeholder metrics object; MAE trainer does not call LatWeightedMetrics."""

    def __call__(self, pred, target):
        return {}


def main(rank: int, world_size: int, conf: dict, backend: str, trial=False):
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    distributed = conf["trainer"]["mode"] in ("fsdp", "ddp")

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

        (thread_workers, train_prefetch, train_pin, train_persist, train_timeout) = _resolve_loader_runtime(conf, "train")
        (valid_workers, valid_prefetch, valid_pin, valid_persist, valid_timeout) = _resolve_loader_runtime(conf, "valid")

        train_filenames = _build_filenames(conf, "train")
        skip_validation = conf["trainer"].get("skip_validation", False)
        valid_filenames = None if skip_validation else _build_filenames(conf, "valid")

        train_dataset, train_sampler = load_dataset_and_sampler(conf, train_filenames, world_size, rank, is_train=True)

        if skip_validation:
            valid_dataset, valid_sampler = None, None
        else:
            valid_dataset, valid_sampler = load_dataset_and_sampler(conf, valid_filenames, world_size, rank, is_train=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=train_pin,
            persistent_workers=train_persist,
            num_workers=thread_workers,
            drop_last=True,
            prefetch_factor=train_prefetch if thread_workers > 0 else None,
            worker_init_fn=_worker_init_fn,
            timeout=train_timeout,
        )

        if skip_validation:
            valid_loader = None
        else:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,
                sampler=valid_sampler,
                pin_memory=valid_pin,
                persistent_workers=valid_persist,
                num_workers=valid_workers,
                drop_last=True,
                prefetch_factor=valid_prefetch if valid_workers > 0 else None,
                worker_init_fn=_worker_init_fn,
                timeout=valid_timeout,
            )

        # Build model, optimizer, scheduler, scaler via shared helper
        m = load_model(conf)
        m.to(device)

        if conf["trainer"].get("compile", False):
            m = torch.compile(m)

        if distributed:
            model = distributed_model_wrapper(conf, m, device)
        else:
            model = m

        conf, model, optimizer, scheduler, scaler = load_model_states_and_optimizer(conf, model, device)

        # MAE trainer owns the loss; pass dummy criterion + metrics
        train_criterion = None
        valid_criterion = None
        metrics = _DummyMetrics()

        trainer = load_trainer(conf)(model, rank)

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


def primary_main():
    parser = ArgumentParser(
        description="Train a WoFS Multi-Modal MAE for self-supervised pre-training."
    )
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("-l", dest="launch", type=int, default=0,
                        help="Submit job to PBS/SLURM scheduler.")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend.",
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

    # Minimal conf setup — skip credit_main_parser to avoid checks designed for
    # the standard WRF forecast pipeline (non-empty variables list, save_loc_dynamic_forcing,
    # etc.) that do not apply to the MAE self-supervised pre-training workflow.
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    conf["data"].setdefault("all_varnames", [])
    conf["data"].setdefault("data_clamp", None)
    conf["data"].setdefault("max_forecast_len", None)
    conf["data"].setdefault("one_shot", None)
    conf["data"].setdefault("total_time_steps", conf["data"].get("forecast_len", 0))
    conf["data"].setdefault("valid_history_len", conf["data"].get("history_len", 1))
    conf["data"].setdefault("valid_forecast_len", conf["data"].get("forecast_len", 0))

    save_loc = conf["save_loc"]
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
