import numpy as np
import torch

from credit.datasets.wrf_wofs_mae import WoFSMAEDataset
from credit.models.wofs_diffmae import WoFSDiffMAE
from credit.transforms.concentration import forward_concentration_transform_numpy, parse_concentration_transform_payload


def _small_model(objective="pred_v"):
    return WoFSDiffMAE(
        modality_channels={
            "background": 2,
            "precip": 3,
            "reflectivity": 1,
            "surface": 1,
            "forcing": 2,
        },
        conditioned_modalities=["background", "surface", "forcing", "reflectivity"],
        target_modality="precip",
        patch_size=4,
        surface_forcing_stride=4,
        image_size=(16, 16),
        embed_dim=32,
        depth=2,
        num_heads=4,
        diffusion={
            "timesteps": 32,
            "sampling_timesteps": 4,
            "objective": objective,
            "beta_schedule": "linear",
            "ddim_sampling_eta": 0.0,
        },
    )


def _cond(batch=2):
    return {
        "background": torch.randn(batch, 2, 16, 16),
        "surface": torch.randn(batch, 1, 16, 16),
        "forcing": torch.randn(batch, 2, 16, 16),
        "reflectivity": torch.randn(batch, 1, 16, 16),
    }


def test_q_sample_matches_closed_form_at_zero_noise():
    model = _small_model()
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.zeros_like(x0)
    t = torch.tensor([0, 10])
    xt = model.q_sample(x0, t, noise)
    expected = model.sqrt_alphas_cumprod[t].view(2, 1, 1, 1) * x0
    assert torch.allclose(xt, expected)


def test_predict_start_noise_round_trip():
    model = _small_model(objective="pred_noise")
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.tensor([5, 20])
    xt = model.q_sample(x0, t, noise)
    recovered = model.predict_start_from_noise(xt, t, noise)
    assert torch.allclose(recovered, x0, atol=1.0e-5, rtol=1.0e-5)


def test_predict_v_round_trip():
    model = _small_model(objective="pred_v")
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    t = torch.tensor([5, 20])
    xt = model.q_sample(x0, t, noise)
    v = model.predict_v(x0, t, noise)
    recovered = model.predict_start_from_v(xt, t, v)
    assert torch.allclose(recovered, x0, atol=1.0e-5, rtol=1.0e-5)


def test_forward_loss_and_sampling_shapes_are_finite():
    model = _small_model()
    precip = torch.randn(2, 3, 16, 16)
    cond = _cond()
    mask = model.random_precip_mask(2, 0.75, precip.device)
    losses = model.p_losses(precip, cond, mask)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    sample = model.sample_precip(cond, mask, sampling_timesteps=2)
    assert sample.shape == precip.shape
    assert torch.isfinite(sample).all()


def test_visible_precip_is_reimposed_during_sampling():
    model = _small_model()
    cond = _cond(batch=1)
    visible = torch.randn(1, 3, 16, 16)
    mask = torch.ones(1, 16)
    mask[:, :4] = 0.0
    sample = model.sample_precip(cond, mask, precip_visible=visible, sampling_timesteps=2)
    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_channel_patch_mask_supports_channel_specific_corrections():
    model = _small_model()
    cond = _cond(batch=1)
    precip = torch.randn(1, 3, 16, 16)
    visible = torch.randn_like(precip)
    mask = torch.zeros(1, 3, 16)
    mask[:, 1, :4] = 1.0

    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    token_mask = model.token_mask_from_precip_mask(mask)
    assert pixel_mask.shape == precip.shape
    assert token_mask.shape == (1, 16)
    assert token_mask[:, :4].eq(1.0).all()
    assert token_mask[:, 4:].eq(0.0).all()

    losses = model.p_losses(precip, cond, mask)
    assert torch.isfinite(losses["loss"])

    sample = model.sample_precip(cond, mask, precip_visible=visible, sampling_timesteps=2)
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_mae_dataset_uses_da_zero_inflated_normalization():
    payload = {
        "transform_type": "zero_inflated_lognormal_probit",
        "zero_floor": 1.0e-11,
        "probit_eps": 1.0e-6,
        "variables": {
            "QRAIN": {
                "clip_max": 1.0,
                "fallback_positive_fit": {"mu": -10.0, "sigma": 1.0},
                "levels": [
                    {"status": "ok", "alpha": 0.9, "mu": -9.0, "sigma": 0.8},
                    {"status": "ok", "alpha": 0.8, "mu": -8.0, "sigma": 1.2},
                    {"status": "ok", "alpha": 0.7, "mu": -7.0, "sigma": 1.5},
                ],
            }
        },
    }
    spec = parse_concentration_transform_payload(payload, variables={"QRAIN"})["QRAIN"]
    ds = WoFSMAEDataset.__new__(WoFSMAEDataset)
    ds.precip_vars = ["QRAIN"]
    ds._mean_values = {"QRAIN": np.array([0.1, -0.2, 0.3], dtype=np.float64)}
    ds._std_values = {"QRAIN": np.array([0.9, 1.1, 1.3], dtype=np.float64)}
    ds._concentration_transform_specs = {"QRAIN": spec}
    ds._concentration_normalization_mode = "zscore"

    raw = np.array(
        [
            [[0.0, 1.0e-8], [2.0e-6, 4.0e-5]],
            [[0.0, 2.0e-8], [3.0e-6, 5.0e-5]],
            [[0.0, 3.0e-8], [4.0e-6, 6.0e-5]],
        ],
        dtype=np.float32,
    )
    out = ds._normalize_array(raw, "QRAIN")
    latent = forward_concentration_transform_numpy(raw, spec, level_axis=0)
    expected = (latent - ds._mean_values["QRAIN"][:, None, None]) / ds._std_values["QRAIN"][:, None, None]
    assert np.allclose(out, expected, atol=1.0e-6, rtol=1.0e-6)
