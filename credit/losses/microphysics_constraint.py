"""
Tier-1 physical consistency losses for microphysics increment learning.

Operates on normalized (z-score) increment tensors.  No physical-space
inverse transform is required — sign monotonicity is preserved by the
concentration transform used in this codebase, and covariance matching is
performed in normalized space where marginal distributions are approximately
Gaussian.

Known (mass mixing ratio, number concentration) pairs targeted by this loss:
  QRAIN   ↔  QNRAIN
  QHAIL   ↔  QNHAIL
  QGRAUP  ↔  QNGRAUPEL
  QSNOW   ↔  QNSNOW

Three loss terms (all Tier-1, normalized space):

  1. sign_consistency  — ReLU penalty on sign-discordant (Δq, Δn) pairs,
                         gated by a magnitude threshold to suppress noise.

  2. corr_alignment    — per-level Pearson correlation of pred aligned to
                         the batch-target correlation.  Uses batch statistics
                         from `target` as the reference so no pre-computed
                         stats are needed.

  3. coral_cov         — 2×2 CORAL covariance matching (pred vs batch-target)
                         per level; Frobenius norm on the difference matrix.

Usage in VariableTotalLoss2D (added via config):

  loss:
    use_microphysics_constraint: true
    microphysics_constraint_weight: 0.05   # scale relative to main loss
    microphysics_constraint:
      sign_weight:    1.0
      corr_weight:    1.0
      coral_weight:   0.5
      sign_threshold: 0.1   # |Δz| below this → sign penalty suppressed
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Maps each mass-mixing-ratio variable to its companion number-concentration variable.
_QN_PAIRS: dict[str, str] = {
    "QRAIN": "QNRAIN",
    "QHAIL": "QNHAIL",
    "QGRAUP": "QNGRAUPEL",
    "QSNOW": "QNSNOW",
}


def _build_pairs(variables: List[str], levels: int) -> List[Tuple[int, int, int, int, str, str]]:
    """Return list of (q_start, q_end, n_start, n_end, q_name, n_name) for each known pair."""
    var_list = list(variables)
    pairs: List[Tuple[int, int, int, int, str, str]] = []
    for idx_q, var_q in enumerate(var_list):
        var_n = _QN_PAIRS.get(var_q)
        if var_n is None or var_n not in var_list:
            continue
        idx_n = var_list.index(var_n)
        pairs.append((
            idx_q * levels,
            idx_q * levels + levels,
            idx_n * levels,
            idx_n * levels + levels,
            var_q,
            var_n,
        ))
    return pairs


def _pearson_batch(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Vectorized per-row Pearson correlation.

    Args:
        a: (L, N)
        b: (L, N)

    Returns:
        rho: (L,) — correlation coefficient per row.
    """
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    num = (a_c * b_c).sum(dim=1)
    denom = (a_c.pow(2).sum(dim=1).sqrt() * b_c.pow(2).sum(dim=1).sqrt()).clamp_min(eps)
    return num / denom


def _cov2x2_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Vectorized per-row 2×2 covariance matrix.

    Args:
        a: (L, N)
        b: (L, N)

    Returns:
        cov: (L, 2, 2)
    """
    n = float(a.shape[1])
    a_c = a - a.mean(dim=1, keepdim=True)
    b_c = b - b.mean(dim=1, keepdim=True)
    cov_aa = (a_c * a_c).sum(dim=1) / n            # (L,)
    cov_bb = (b_c * b_c).sum(dim=1) / n            # (L,)
    cov_ab = (a_c * b_c).sum(dim=1) / n            # (L,)
    row0 = torch.stack([cov_aa, cov_ab], dim=1)    # (L, 2)
    row1 = torch.stack([cov_ab, cov_bb], dim=1)    # (L, 2)
    return torch.stack([row0, row1], dim=1)         # (L, 2, 2)


class MicrophysicsConsistencyLoss(nn.Module):
    """
    Tier-1 microphysics physical consistency auxiliary loss.

    Inputs are expected in normalized z-score space with shape
    ``(B, total_channels, time, H, W)``.  Channel order follows the
    convention used in ``VariableTotalLoss2D``:

        channels = [var0_lev0, ..., var0_levL, var1_lev0, ..., varN_levL]

    Args:
        variables      : ordered list of prognostic variable names, matching
                         ``conf["data"]["variables"]``.
        levels         : number of vertical levels per variable.
        sign_weight    : scalar weight for the sign-consistency term.
        corr_weight    : scalar weight for per-level correlation alignment.
        coral_weight   : scalar weight for 2×2 CORAL covariance matching.
        sign_threshold : |Δz| magnitude threshold below which the sign
                         penalty is suppressed (default 0.1).

    Notes:
        All three terms use batch statistics from ``target`` as the reference
        distribution for the correlation and CORAL terms.  No pre-computed
        global statistics are required.  A separate script
        (``scripts/compute_qn_target_statistics.py``) can be used to
        compute global statistics if fixed targets are preferred.
    """

    def __init__(
        self,
        variables: List[str],
        levels: int,
        sign_weight: float = 1.0,
        corr_weight: float = 1.0,
        coral_weight: float = 0.5,
        sign_threshold: float = 0.1,
    ) -> None:
        super().__init__()

        self.sign_weight = float(sign_weight)
        self.corr_weight = float(corr_weight)
        self.coral_weight = float(coral_weight)
        self.sign_threshold = float(sign_threshold)
        self.levels = int(levels)

        self.pairs = _build_pairs(variables, self.levels)

        if not self.pairs:
            logger.warning(
                "MicrophysicsConsistencyLoss: no known Q/N variable pairs found in %s.  "
                "Loss will return zero.",
                variables,
            )
        else:
            pair_names = [(q, n) for *_, q, n in self.pairs]
            logger.info("MicrophysicsConsistencyLoss: active pairs = %s", pair_names)

    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the microphysics constraint loss.

        Args:
            pred   : predicted increment tensor  ``(B, C, T, H, W)`` — normalized.
            target : true increment tensor        ``(B, C, T, H, W)`` — normalized.
                     Used as the reference distribution for correlation and
                     CORAL terms.  Not compared element-wise.

        Returns:
            Scalar loss (mean over all active pairs and loss terms).
        """
        if not self.pairs:
            return pred.new_zeros(())

        pred = pred.float()
        target = target.float()

        L = self.levels
        sign_total = pred.new_zeros(())
        corr_total = pred.new_zeros(())
        coral_total = pred.new_zeros(())

        for qs, qe, ns, ne, _qname, _nname in self.pairs:
            # ── extract and flatten to (L, N) where N = B * T * H * W ──
            # pred shape: (B, C, T, H, W)
            dq_p = pred[:, qs:qe, :, :, :].permute(1, 0, 2, 3, 4).reshape(L, -1)   # (L, N)
            dn_p = pred[:, ns:ne, :, :, :].permute(1, 0, 2, 3, 4).reshape(L, -1)   # (L, N)
            dq_t = target[:, qs:qe, :, :, :].permute(1, 0, 2, 3, 4).reshape(L, -1) # (L, N)
            dn_t = target[:, ns:ne, :, :, :].permute(1, 0, 2, 3, 4).reshape(L, -1) # (L, N)

            # ── 1. Sign-consistency regularization ───────────────────
            if self.sign_weight > 0.0:
                # Penalise sign-discordant pairs; weight by the smaller magnitude
                # of the two increments so clear-air (near-zero) is suppressed.
                mag = torch.minimum(dq_p.abs(), dn_p.abs())                # (L, N)
                gate = torch.relu(mag - self.sign_threshold)               # (L, N)
                penalty = torch.relu(-dq_p * dn_p)                        # (L, N)
                denom = gate.sum().clamp_min(1.0)
                sign_total = sign_total + (penalty * gate).sum() / denom

            # ── 2. Per-level Pearson correlation alignment ────────────
            if self.corr_weight > 0.0:
                rho_pred = _pearson_batch(dq_p, dn_p)   # (L,)
                rho_true = _pearson_batch(dq_t, dn_t)   # (L,)
                corr_total = corr_total + (rho_pred - rho_true).pow(2).mean()

            # ── 3. 2×2 CORAL covariance matching ─────────────────────
            if self.coral_weight > 0.0:
                cov_pred = _cov2x2_batch(dq_p, dn_p)    # (L, 2, 2)
                cov_true = _cov2x2_batch(dq_t, dn_t)    # (L, 2, 2)
                # Mean Frobenius norm over levels; divide by 4 (matrix entries)
                coral_total = coral_total + (cov_pred - cov_true).pow(2).mean()

        n_pairs = float(len(self.pairs))
        loss = (
            self.sign_weight  * sign_total  / n_pairs
            + self.corr_weight  * corr_total  / n_pairs
            + self.coral_weight * coral_total / n_pairs
        )
        return loss
