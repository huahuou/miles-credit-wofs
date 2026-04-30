#!/usr/bin/env python
"""
compute_qn_target_statistics.py
--------------------------------
Pre-compute per-level Q–N joint statistics (Pearson correlation and 2×2
covariance matrix) of normalized increment pairs from training data.

The results can be used as fixed-target reference distributions for the
Tier-1 microphysics constraint loss in
``credit/losses/microphysics_constraint.py``.  Without running this script
the loss already works by matching batch statistics from ``target``, so this
script is optional but useful for:

  - Diagnosing the Q–N correlation structure of the training data.
  - Providing fixed reference stats (rather than per-batch target stats) to
    the loss if desired in future iterations.

Outputs (saved to ``--out_dir``):
  target_corr.npz   — shape (n_pairs, L) — per-pair per-level Pearson ρ
  target_cov.npz    — shape (n_pairs, L, 2, 2) — per-pair per-level 2×2 cov
  pair_names.json   — list of [q_name, n_name] strings, same order as arrays

Usage:
  python scripts/compute_qn_target_statistics.py \
      -c config/wofs_credit_wrf_da_increment.yml \
      --out_dir /work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats \
      --max_files 200        # optional: limit number of training zarr files
      --batch_size 16

Cluster (CPU node):
  srun -A gpu-ai4wp -p u1-compute --mem=128g -N 1 -t 1:30:00 --pty bash -il
  module load rdhpcs-conda && module load cuda/12.8 && conda activate credit-wofs
  python scripts/compute_qn_target_statistics.py -c config/wofs_credit_wrf_da_increment.yml \\
      --out_dir /work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add the repo root and applications/ to path so credit imports work when run from scripts/
_repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "applications"))

from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset
from credit.parser import credit_main_parser
from applications.train_wrf_wofs_da import _build_params, _select_files, _sync_prognostic_levels

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("compute_qn_stats")

# Known (mass, number) variable pairs — must match microphysics_constraint.py
_QN_PAIRS: dict[str, str] = {
    "QRAIN": "QNRAIN",
    "QHAIL": "QNHAIL",
    "QGRAUP": "QNGRAUPEL",
    "QSNOW": "QNSNOW",
}


def _build_pairs_from_vars(variables: list[str], levels: int):
    """Return list of (qs, qe, ns, ne, q_name, n_name) channel-index tuples."""
    pairs = []
    for idx_q, var_q in enumerate(variables):
        var_n = _QN_PAIRS.get(var_q)
        if var_n is None or var_n not in variables:
            continue
        idx_n = variables.index(var_n)
        pairs.append((
            idx_q * levels, idx_q * levels + levels,
            idx_n * levels, idx_n * levels + levels,
            var_q, var_n,
        ))
    return pairs


class _OnlineStats:
    """
    Online accumulation of first/second-order statistics for a pair (q, n)
    across all (B, T, H, W) samples, kept per vertical level L.

    Uses the parallel / two-pass formulation to remain numerically stable
    without storing all values.
    """

    def __init__(self, levels: int, device: torch.device):
        self.L = levels
        self.device = device
        self.reset()

    def reset(self):
        L = self.L
        self.count = torch.zeros(L, dtype=torch.float64, device=self.device)
        self.sum_q = torch.zeros(L, dtype=torch.float64, device=self.device)
        self.sum_n = torch.zeros(L, dtype=torch.float64, device=self.device)
        self.sum_q2 = torch.zeros(L, dtype=torch.float64, device=self.device)
        self.sum_n2 = torch.zeros(L, dtype=torch.float64, device=self.device)
        self.sum_qn = torch.zeros(L, dtype=torch.float64, device=self.device)

    def update(self, dq: torch.Tensor, dn: torch.Tensor):
        """
        Args:
            dq: (L, N) — QRAIN normalized increments for a batch
            dn: (L, N) — QNRAIN normalized increments for the same batch
        """
        dq = dq.to(dtype=torch.float64, device=self.device)
        dn = dn.to(dtype=torch.float64, device=self.device)
        n = dq.shape[1]
        self.count += n
        self.sum_q += dq.sum(dim=1)
        self.sum_n += dn.sum(dim=1)
        self.sum_q2 += (dq * dq).sum(dim=1)
        self.sum_n2 += (dn * dn).sum(dim=1)
        self.sum_qn += (dq * dn).sum(dim=1)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            corr : (L,)    — per-level Pearson correlation coefficient
            cov  : (L,2,2) — per-level 2×2 covariance matrix
        """
        n = self.count.clamp_min(1.0)
        mean_q = self.sum_q / n
        mean_n = self.sum_n / n

        var_q = self.sum_q2 / n - mean_q ** 2
        var_n = self.sum_n2 / n - mean_n ** 2
        cov_qn = self.sum_qn / n - mean_q * mean_n

        std_q = var_q.clamp_min(0.0).sqrt()
        std_n = var_n.clamp_min(0.0).sqrt()
        eps = torch.tensor(1e-8, dtype=torch.float64, device=self.device)
        corr = cov_qn / (std_q * std_n + eps)
        corr = corr.clamp(-1.0, 1.0)

        # 2×2 covariance matrix per level: [[var_q, cov_qn],[cov_qn, var_n]]
        L = self.L
        cov = torch.zeros(L, 2, 2, dtype=torch.float64, device=self.device)
        cov[:, 0, 0] = var_q
        cov[:, 1, 1] = var_n
        cov[:, 0, 1] = cov_qn
        cov[:, 1, 0] = cov_qn

        return corr.cpu().numpy(), cov.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Compute Q–N joint statistics from WoFS DA training data")
    parser.add_argument("-c", "--config", dest="model_config", required=True)
    parser.add_argument("--out_dir", default=None, help="Output directory for stats files")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max training zarr files to process (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_glob", default=None,
                        help="Override config train data glob (e.g. /scratch5/.../cases/wofs_*.zarr.zip)")
    args = parser.parse_args()

    with open(args.model_config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    _sync_prognostic_levels(conf)

    out_dir = args.out_dir or os.path.dirname(args.model_config)
    os.makedirs(out_dir, exist_ok=True)

    variables = conf["data"]["variables"]
    levels = int(conf["data"]["prognostic_levels"])

    pairs = _build_pairs_from_vars(variables, levels)
    if not pairs:
        logger.error("No known Q/N pairs in variables list %s — nothing to compute.", variables)
        sys.exit(1)

    logger.info("Active pairs: %s", [(q, n) for *_, q, n in pairs])

    # ── Override data paths if --data_glob was given ──────────────────────────
    if args.data_glob is not None:
        for key in ("save_loc", "save_loc_surface", "save_loc_dynamic_forcing"):
            if conf["data"].get(key):
                conf["data"][key] = args.data_glob
        # Derive stats dir: parent of the cases/ directory
        _data_dir = os.path.dirname(args.data_glob.split("*")[0].rstrip("/"))
        _base = os.path.dirname(_data_dir) if os.path.basename(_data_dir) == "cases" else _data_dir
        _stats_dir = os.path.join(_base, "stats")
        for stat_key, fname in (("mean_path", "mean.nc"), ("std_path", "std.nc"),
                                 ("concentration_params_json", "concentration_tuning.json")):
            candidate = os.path.join(_stats_dir, fname)
            if conf["data"].get(stat_key) and os.path.exists(candidate):
                conf["data"][stat_key] = candidate
        logger.info("--data_glob override applied to conf['data'] paths: %s (stats: %s)",
                    args.data_glob, _stats_dir)

    # ── Build training dataset (subset of files if --max_files) ──────────────
    param_interior, param_outside = _build_params(conf, "train")

    if args.max_files > 0 and len(param_interior["filenames"]) > args.max_files:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(param_interior["filenames"]), size=args.max_files, replace=False)
        param_interior["filenames"] = [param_interior["filenames"][i] for i in sorted(indices)]
        if param_interior.get("filename_dyn_forcing"):
            param_interior["filename_dyn_forcing"] = [
                param_interior["filename_dyn_forcing"][i] for i in sorted(indices)
            ]
        logger.info("Using %d / total training files (--max_files).", args.max_files)

    logger.info("Building dataset with %d files ...", len(param_interior["filenames"]))
    dataset = WoFSDAIncrementDataset(
        param_interior,
        param_outside,
        conf=conf,
        seed=42,
    )
    # Force index build (single-process, no workers needed for that)
    dataset._open_datasets()

    logger.info("Dataset length: %d samples", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = torch.device("cpu")
    accumulators = [_OnlineStats(levels, device) for _ in pairs]

    from tqdm import tqdm
    n_batches = len(loader)
    logger.info("Iterating %d batches (batch_size=%d) ...", n_batches, args.batch_size)

    for batch_idx, batch in enumerate(tqdm(loader, total=n_batches, desc="accumulating stats")):
        # y shape from dataset: (B, time=1, n_prog_vars, levels, H, W)
        # After DataLoader collation: (B, 1, 8, 17, 300, 300)
        y_raw = batch["y"]   # (B, time, vars, levels, H, W)

        # Flatten to (B, vars*levels, 1, H, W) matching reshape_only convention
        B, T, V, L, H, W = y_raw.shape
        # reshape: (B, vars*levels, T, H, W)
        y = y_raw.permute(0, 2, 3, 1, 4, 5).reshape(B, V * L, T, H, W)

        for acc, (qs, qe, ns, ne, _, _) in zip(accumulators, pairs):
            # Extract pair channels: (B, L, T, H, W) → permute level to front → (L, N)
            dq = y[:, qs:qe, :, :, :].permute(1, 0, 2, 3, 4).reshape(levels, -1)
            dn = y[:, ns:ne, :, :, :].permute(1, 0, 2, 3, 4).reshape(levels, -1)
            acc.update(dq, dn)

        if (batch_idx + 1) % 500 == 0:
            logger.info("  processed %d / %d batches", batch_idx + 1, n_batches)

    # ── Finalize and save ──────────────────────────────────────────────────────
    all_corr = []
    all_cov = []
    pair_names = []
    for acc, (_, _, _, _, q_name, n_name) in zip(accumulators, pairs):
        corr, cov = acc.finalize()
        all_corr.append(corr)
        all_cov.append(cov)
        pair_names.append([q_name, n_name])
        logger.info(
            "  %s/%s — mean Pearson ρ across levels: %.4f  (min=%.4f, max=%.4f)",
            q_name, n_name, float(corr.mean()), float(corr.min()), float(corr.max()),
        )
        logger.info(
            "  %s/%s — mean off-diagonal cov: %.4e  var_Q: %.4e  var_N: %.4e",
            q_name, n_name,
            float(cov[:, 0, 1].mean()),
            float(cov[:, 0, 0].mean()),
            float(cov[:, 1, 1].mean()),
        )

    corr_arr = np.stack(all_corr, axis=0)   # (n_pairs, L)
    cov_arr  = np.stack(all_cov,  axis=0)   # (n_pairs, L, 2, 2)

    corr_path = os.path.join(out_dir, "target_qn_corr.npz")
    cov_path  = os.path.join(out_dir, "target_qn_cov.npz")
    names_path = os.path.join(out_dir, "target_qn_pair_names.json")

    np.savez(corr_path, corr=corr_arr)
    np.savez(cov_path,  cov=cov_arr)
    with open(names_path, "w") as f:
        json.dump(pair_names, f, indent=2)

    logger.info("Saved:\n  %s\n  %s\n  %s", corr_path, cov_path, names_path)
    logger.info(
        "To use fixed global stats in the loss, add to config under loss.microphysics_constraint:\n"
        "  target_corr_path: %s\n"
        "  target_cov_path:  %s",
        corr_path, cov_path,
    )


if __name__ == "__main__":
    main()
