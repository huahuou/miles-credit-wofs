import logging
import os
import re
import shutil
import sys
import warnings
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import optuna
import torch
import yaml
from echo.src.base_objective import BaseObjective
from torch.utils.data.distributed import DistributedSampler

from train_wrf import load_model_states_and_optimizer
from credit.datasets.wrf_wofs_multistep import WoFSMultiStep
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.losses.weighted_loss import VariableTotalLoss2D
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.parser import credit_main_parser
from credit.pbs import launch_script, launch_script_mpi
from credit.scheduler import annealed_probability
from credit.seed import seed_everything
from credit.trainers import load_trainer
from credit.transforms import load_transforms

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _worker_init_fn(worker_id):
    """
    DataLoader worker init: if the dataset exposes _open_datasets, call it here
    so each worker opens xarray/zarr resources once in-worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    ds = worker_info.dataset
    if hasattr(ds, "_open_datasets") and callable(ds._open_datasets):
        try:
            ds._open_datasets()
        except Exception as e:
            logging.warning(f"worker_init_fn failed in worker {worker_id}: {e}")
            raise


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr$", Path(file_path).name)
    return match.group(1) if match else None


def _select_files(
    glob_pattern: str,
    years_range: list[int] | list[str] | None,
    date_range: list[str] | tuple[str, str] | None = None,
) -> list[str]:
    files = sorted(glob(glob_pattern))

    if date_range is not None:
        start_date, end_date = str(date_range[0]), str(date_range[1])
        selected = []
        for file_path in files:
            case_date = _extract_case_date(file_path)
            if case_date is not None and start_date <= case_date <= end_date:
                selected.append(file_path)
        return selected

    if years_range is None:
        return files

    years = [str(year) for year in range(int(years_range[0]), int(years_range[1]))]
    return [file for file in files if any(year in file for year in years)]


def load_dataset_and_sampler(conf, param_interior, param_outside, world_size, rank, is_train=True):
    seed = conf["seed"]
    transforms = load_transforms(conf)

    dataset = WoFSMultiStep(
        param_interior,
        param_outside,
        transform=transforms,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=is_train,
        drop_last=True,
    )

    logging.info("Loaded WoFS multi-step dataset and distributed sampler")
    return dataset, sampler


def _build_params(conf, split: str):
    years_range = conf["data"]["train_years"] if split == "train" else conf["data"]["valid_years"]
    date_range = conf["data"].get("train_date_range") if split == "train" else conf["data"].get("valid_date_range")
    history_key = "history_len" if split == "train" else "valid_history_len"
    forecast_key = "forecast_len" if split == "train" else "valid_forecast_len"

    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_surface": conf["data"]["surface_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "varname_forcing": conf["data"]["forcing_variables"],
        "varname_static": conf["data"]["static_variables"],
        "varname_diagnostic": conf["data"]["diagnostic_variables"],
        "filenames": _select_files(conf["data"]["save_loc"], years_range, date_range),
        "filename_surface": _select_files(conf["data"]["save_loc_surface"], years_range, date_range)
        if conf["data"].get("save_loc_surface")
        else None,
        "filename_dyn_forcing": _select_files(conf["data"]["save_loc_dynamic_forcing"], years_range, date_range)
        if conf["data"].get("save_loc_dynamic_forcing")
        else None,
        "filename_forcing": conf["data"].get("save_loc_forcing"),
        "filename_static": conf["data"].get("save_loc_static"),
        "filename_diagnostic": _select_files(conf["data"]["save_loc_diagnostic"], years_range, date_range)
        if conf["data"].get("save_loc_diagnostic")
        else None,
        "history_len": conf["data"][history_key],
        "forecast_len": conf["data"][forecast_key],
        "zarr_time_chunk": conf["data"].get("zarr_time_chunk", 0),
    }

    param_outside = {
        "varname_upper_air": conf["data"]["boundary"]["variables"],
        "varname_surface": conf["data"]["boundary"].get("surface_variables", []),
        "history_len": conf["data"]["boundary"]["history_len"],
        "forecast_len": conf["data"]["boundary"]["forecast_len"],
    }

    return param_interior, param_outside


def main(rank, world_size, conf, backend, trial=False):
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    seed_everything(conf["seed"])

    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    thread_workers = conf["trainer"].get("thread_workers", 0)
    valid_thread_workers = conf["trainer"].get("valid_thread_workers", thread_workers)
    train_prefetch_factor = conf["trainer"].get("prefetch_factor", 4)
    valid_prefetch_factor = conf["trainer"].get("valid_prefetch_factor", train_prefetch_factor)

    if conf["data"]["scaler_type"] != "std-wrf":
        raise ValueError("WoFS multi-step training app currently supports scaler_type='std-wrf' only")

    param_interior_train, param_outside_train = _build_params(conf, "train")
    param_interior_valid, param_outside_valid = _build_params(conf, "valid")

    train_dataset, train_sampler = load_dataset_and_sampler(
        conf,
        param_interior_train,
        param_outside_train,
        world_size,
        rank,
        is_train=True,
    )

    valid_dataset, valid_sampler = load_dataset_and_sampler(
        conf,
        param_interior_valid,
        param_outside_valid,
        world_size,
        rank,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=True,
        prefetch_factor=train_prefetch_factor if thread_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=True,
        persistent_workers=True if valid_thread_workers > 0 else False,
        num_workers=valid_thread_workers,
        drop_last=True,
        prefetch_factor=valid_prefetch_factor if valid_thread_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )

    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    if conf["trainer"]["mode"] in ["ddp", "fsdp"]:
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


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        conf["model"]["dim_head"] = conf["model"]["dim"]
        conf["model"]["vq_codebook_dim"] = conf["model"]["dim"]
        try:
            return main(0, 1, conf, backend="nccl", trial=trial)
        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                logging.warning(f"Pruning trial {trial.number} due to runtime error: {str(E)}.")
                raise optuna.TrialPruned()
            raise E


def primary_main():
    parser = ArgumentParser(description="Train a WoFS multi-step WRF model with same-file t0 conditioning")
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

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)

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