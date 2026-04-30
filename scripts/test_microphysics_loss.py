"""
Smoke-test for MicrophysicsConsistencyLoss and its integration into VariableTotalLoss2D.
Run with:
    cd /home/Zhanxiang.Hua/miles-credit-wofs
    python scripts/test_microphysics_loss.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label, expr):
    if expr:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: direct import and forward pass
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: MicrophysicsConsistencyLoss standalone ===")
try:
    from credit.losses.microphysics_constraint import MicrophysicsConsistencyLoss, _build_pairs

    variables = ["QRAIN", "QNRAIN", "QHAIL", "QNHAIL", "QGRAUP", "QNGRAUPEL", "QSNOW", "QNSNOW"]
    levels = 17

    pairs = _build_pairs(variables, levels)
    pair_names = [(q, n) for *_, q, n in pairs]
    check(f"4 pairs detected: {pair_names}", len(pairs) == 4)

    fn = MicrophysicsConsistencyLoss(variables, levels)
    B, C, T, H, W = 3, len(variables) * levels, 1, 20, 20
    pred   = torch.randn(B, C, T, H, W, requires_grad=True)
    target = torch.randn(B, C, T, H, W)

    loss = fn(pred, target)
    check("Loss is scalar", loss.ndim == 0)
    check("Loss is finite", torch.isfinite(loss).item())
    check("Loss has grad_fn", loss.grad_fn is not None)

    loss.backward()
    grad = pred.grad
    check("Gradient flows back to pred", grad is not None)
    check("Gradient is finite", torch.isfinite(grad).all().item())
    print(f"  Loss value: {loss.item():.6f}")

except Exception:
    print(f"  {FAIL}  Exception raised:")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: sign-consistency term is zero when all increments are concordant
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: Sign-consistency — zero when all same sign ===")
try:
    from credit.losses.microphysics_constraint import MicrophysicsConsistencyLoss

    variables = ["QRAIN", "QNRAIN"]
    levels = 3
    fn = MicrophysicsConsistencyLoss(variables, levels,
                                      sign_weight=1.0, corr_weight=0.0, coral_weight=0.0,
                                      sign_threshold=0.0)

    # All positive increments — sign penalty should be zero
    B, C, T, H, W = 2, 6, 1, 10, 10
    pos = torch.abs(torch.randn(B, C, T, H, W)) + 0.5  # strictly positive
    loss_zero = fn(pos, pos)
    check("Sign loss == 0 when all increments concordant positive", loss_zero.item() < 1e-6)

    # Mix: QRAIN positive, QNRAIN negative above threshold → should be nonzero
    mixed = pos.clone()
    mixed[:, levels:, :, :, :] = -mixed[:, levels:, :, :, :]  # flip QNRAIN sign
    loss_mixed = fn(mixed, mixed)
    check("Sign loss > 0 when increments are discordant", loss_mixed.item() > 0.0)

except Exception:
    print(f"  {FAIL}  Exception raised:")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: CORAL term zero when pred == target (same statistics)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: CORAL — near zero when pred matches target stats ===")
try:
    from credit.losses.microphysics_constraint import MicrophysicsConsistencyLoss

    variables = ["QRAIN", "QNRAIN"]
    levels = 3
    fn = MicrophysicsConsistencyLoss(variables, levels,
                                      sign_weight=0.0, corr_weight=0.0, coral_weight=1.0)
    B, C, T, H, W = 8, 6, 1, 50, 50
    x = torch.randn(B, C, T, H, W)
    loss_identical = fn(x, x)
    check("CORAL loss == 0 when pred == target", loss_identical.item() < 1e-4)

    y = torch.randn(B, C, T, H, W) * 5.0  # very different distribution
    loss_diff = fn(x, y)
    check("CORAL loss > 0 when pred != target", loss_diff.item() > 1e-4)

except Exception:
    print(f"  {FAIL}  Exception raised:")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: VariableTotalLoss2D integration with use_microphysics_constraint=true
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: VariableTotalLoss2D integration ===")
try:
    from credit.losses.weighted_loss import VariableTotalLoss2D

    # Minimal config skeleton
    conf = {
        "data": {
            "variables": ["QRAIN", "QNRAIN", "QHAIL", "QNHAIL", "QGRAUP", "QNGRAUPEL", "QSNOW", "QNSNOW"],
            "surface_variables": [],
            "diagnostic_variables": [],
        },
        "loss": {
            "training_loss": "mse",
            "use_latitude_weights": False,
            "use_variable_weights": False,
            "use_spectral_loss": False,
            "use_power_loss": False,
            "use_microphysics_constraint": True,
            "microphysics_constraint_weight": 0.05,
            "microphysics_constraint": {
                "sign_weight": 1.0,
                "corr_weight": 1.0,
                "coral_weight": 0.5,
                "sign_threshold": 0.1,
            },
        },
        "model": {"levels": 17},
    }

    train_criterion = VariableTotalLoss2D(conf, validation=False)
    valid_criterion = VariableTotalLoss2D(conf, validation=True)

    check("train_criterion has micro_constraint_loss", hasattr(train_criterion, "micro_constraint_loss"))
    check("valid_criterion does NOT have micro constraint active",
          not train_criterion.validation and
          not valid_criterion.use_microphysics_constraint)

    # Shape: (B, total_channels, H, W) — VariableTotalLoss2D gets 4D tensors
    # Variables: 8 vars * 17 levels = 136 channels
    B, C, H, W = 3, 8 * 17, 20, 20
    # The trainer passes (B, C, T, H, W) shape but VariableTotalLoss2D iterates
    # over dim=1 per-channel — let's confirm the forward signature.
    # Looking at the trainer: y shape after reshape_only is (B, C, T, H, W)
    # and the loss operates on that. The micro constraint also expects (B, C, T, H, W).
    T = 1
    pred   = torch.randn(B, C, T, H, W, requires_grad=True)
    target = torch.randn(B, C, T, H, W)

    loss_train = train_criterion(target, pred)
    check("Training loss is scalar and finite", loss_train.ndim == 0 and torch.isfinite(loss_train))
    loss_train.backward()
    check("Gradient flows through combined loss", pred.grad is not None)

    pred2 = torch.randn(B, C, T, H, W)
    loss_valid = valid_criterion(target, pred2)
    check("Validation loss is scalar and finite", loss_valid.ndim == 0 and torch.isfinite(loss_valid))

    # Confirm micro constraint actually raises the training loss vs baseline
    fn_no_micro = VariableTotalLoss2D({
        "data": conf["data"],
        "loss": {**conf["loss"], "use_microphysics_constraint": False},
        "model": conf["model"],
    }, validation=False)
    # Use identical pred/target so MSE is the same
    p = torch.randn(B, C, T, H, W)
    t = torch.randn(B, C, T, H, W)
    loss_with    = train_criterion(t, p).item()
    loss_without = fn_no_micro(t, p).item()
    print(f"  Loss WITH  constraint: {loss_with:.6f}")
    print(f"  Loss WITHOUT constraint: {loss_without:.6f}")
    # They can be equal if the batch happens to have perfect covariance match,
    # but the sign+CORAL terms usually add something positive.
    check("Micro constraint adds non-negative loss component",
          loss_with >= loss_without - 1e-5)

except Exception:
    print(f"  {FAIL}  Exception raised:")
    traceback.print_exc()


print("\n=== All tests complete ===\n")
