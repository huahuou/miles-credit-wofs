import numpy as np
import torch

from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset
from credit.transforms.concentration import (
    inverse_concentration_transform_numpy,
    inverse_concentration_transform_torch,
    parse_concentration_transform_payload,
    forward_concentration_transform_numpy,
)


def _build_zero_inflated_spec():
    payload = {
        "transform_type": "zero_inflated_lognormal_probit",
        "zero_floor": 1.0e-11,
        "probit_eps": 1.0e-6,
        "variables": {
            "QRAIN": {
                "clip_max": 1.0,
                "fallback_positive_fit": {"mu": -10.0, "sigma": 1.0},
                "levels": [
                    {"status": "ok", "alpha": 0.92, "mu": -9.5, "sigma": 0.8},
                    {"status": "borrow_fallback", "alpha": 0.995, "mu": -10.0, "sigma": 1.0},
                    {"status": "degenerate_zero", "alpha": 1.0},
                ],
            }
        },
    }
    return parse_concentration_transform_payload(payload, variables={"QRAIN"})["QRAIN"]


def test_zero_inflated_round_trip_numpy():
    spec = _build_zero_inflated_spec()
    raw = np.array(
        [
            [[0.0, 1.0e-12], [5.0e-10, 1.0e-6]],
            [[0.0, 1.0e-7], [2.0e-5, 3.0e-4]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    latent = forward_concentration_transform_numpy(raw, spec, level_axis=0)
    recovered = inverse_concentration_transform_numpy(latent, spec, level_axis=0)
    expected = np.where(raw < spec["zero_floor"], 0.0, np.clip(raw, 0.0, spec["clip_max"]))
    assert np.allclose(recovered, expected, atol=1.0e-8, rtol=1.0e-6)


def test_zero_inflated_degenerate_level_stays_zero():
    spec = _build_zero_inflated_spec()
    latent = np.linspace(-8.0, 8.0, 25, dtype=np.float64).reshape(1, 5, 5)
    recovered = inverse_concentration_transform_numpy(latent, spec, level_axis=0)
    assert np.allclose(recovered, 0.0)


def test_zero_inflated_torch_inverse_matches_numpy():
    spec = _build_zero_inflated_spec()
    latent = np.linspace(-4.0, 4.0, 3 * 6 * 7, dtype=np.float64).reshape(3, 6, 7)
    out_np = inverse_concentration_transform_numpy(latent, spec, level_axis=0)
    out_torch = inverse_concentration_transform_torch(torch.from_numpy(latent).float(), spec, level_axis=0)
    assert np.allclose(out_torch.numpy(), out_np, atol=5.0e-6, rtol=5.0e-5)


def test_dataset_denormalize_increment_round_trip_with_zero_inflated_transform():
    spec = _build_zero_inflated_spec()
    ds = WoFSDAIncrementDataset.__new__(WoFSDAIncrementDataset)
    ds._mean_values = {"QRAIN": np.array([0.1, -0.2, 0.0], dtype=np.float64)}
    ds._std_values = {"QRAIN": np.array([0.9, 1.1, 1.0], dtype=np.float64)}
    ds._concentration_transform_specs = {"QRAIN": spec}
    ds._log_transform_params = {}
    ds._concentration_params = {}
    ds._prognostic_levels = 3
    ds.varname_prognostic = ["QRAIN"]

    raw_t0 = np.array(
        [
            [[0.0, 1.0e-8], [3.0e-7, 5.0e-6]],
            [[0.0, 2.0e-7], [1.0e-5, 6.0e-5]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    raw_t1 = np.array(
        [
            [[0.0, 5.0e-8], [5.0e-7, 7.5e-6]],
            [[0.0, 3.0e-7], [2.5e-5, 1.2e-4]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    latent_t0 = forward_concentration_transform_numpy(raw_t0, spec, level_axis=0)
    latent_t1 = forward_concentration_transform_numpy(raw_t1, spec, level_axis=0)
    mean = ds._mean_values["QRAIN"][:, None, None]
    std = ds._std_values["QRAIN"][:, None, None]
    z0 = (latent_t0 - mean) / std
    z1 = (latent_t1 - mean) / std
    normalized_increment = z1 - z0

    increment = ds.denormalize_increment(normalized_increment, raw_t0, "QRAIN")
    assert np.allclose(raw_t0 + increment, raw_t1, atol=1.0e-7, rtol=1.0e-5)
