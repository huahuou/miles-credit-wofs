import numpy as np
import torch

from credit.datasets.wrf_wofs_mae import WoFSMAEDataset
from credit.models.wofs_diffmae import WoFSDiffMAE
from credit.trainers.trainerWRF_diffmae import TrainerDiffMAE
from credit.transforms.concentration import forward_concentration_transform_numpy, parse_concentration_transform_payload
from applications.rollout_wrf_wofs_mae_da import _build_condition_dict, _sample_mask
from applications.rollout_wrf_wofs_mae_da_metrics import (
    _grouped_mask_to_runtime_mask,
    _grouped_patch_to_pixel_mask,
    compute_masked_normalized_metrics,
    compute_masked_physical_metrics,
)
from credit.wofs_diffmae_mask_utils import build_grouped_patch_masks, grouped_patch_mask_time_slice, load_mask_bundle, save_mask_bundle


def _small_model(objective="pred_v", decoder_type="concat_self"):
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
        decoder_type=decoder_type,
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


def _small_grouped_model(
    objective="pred_v",
    decoder_type="concat_self",
    grouped_decoder_scope="per_group",
    target_attention_window_size=0,
    anti_patch_refiner=None,
):
    return WoFSDiffMAE(
        modality_channels={
            "background": 2,
            "precip": 6,
            "reflectivity": 1,
            "surface": 1,
            "forcing": 2,
        },
        conditioned_modalities=["background", "surface", "forcing", "reflectivity"],
        target_modality="precip",
        precip_grouping="grouped",
        decoder_type=decoder_type,
        grouped_decoder_scope=grouped_decoder_scope,
        precip_group_names=["rain", "hail", "snow"],
        precip_group_channels=[2, 2, 2],
        patch_size=4,
        surface_forcing_stride=4,
        image_size=(16, 16),
        embed_dim=32,
        depth=2,
        num_heads=4,
        target_attention_window_size=target_attention_window_size,
        anti_patch_refiner=anti_patch_refiner,
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


def _cond_grouped(batch=2):
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


def test_ddim_update_matches_reference_formula():
    model = _small_model(objective="pred_noise")
    x_t = torch.randn(2, 3, 16, 16)
    eps = torch.randn_like(x_t)
    time = 20
    time_next = 10
    eta = 0.0

    alpha = model.alphas_cumprod[time]
    alpha_next = model.alphas_cumprod[time_next]
    x_start = model.predict_start_from_noise(
        x_t,
        torch.full((x_t.shape[0],), time, dtype=torch.long),
        eps,
    )
    ours = x_start * alpha_next.sqrt() + (1 - alpha_next).sqrt() * eps
    reference = (
        torch.sqrt(alpha_next / alpha) * x_t
        + (torch.sqrt(1 - alpha_next) - torch.sqrt((alpha_next * (1 - alpha)) / alpha)) * eps
    )

    assert eta == 0.0
    assert torch.allclose(ours, reference, atol=1.0e-5, rtol=1.0e-5)


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


def test_cross_self_decoder_forward_loss_and_sampling_shapes_are_finite():
    model = _small_model(decoder_type="cross_self")
    precip = torch.randn(2, 3, 16, 16)
    cond = _cond()
    mask = model.random_precip_mask(2, 0.75, precip.device)
    losses = model.p_losses(precip, cond, mask)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    sample = model.sample_precip(cond, mask, sampling_timesteps=2)
    assert sample.shape == precip.shape
    assert torch.isfinite(sample).all()


def test_masked_noisy_state_affects_prediction():
    model = _small_model()
    cond = _cond(batch=1)
    t = torch.tensor([5])
    mask = torch.ones(1, 16)
    x1 = torch.randn(1, 3, 16, 16)
    x2 = x1 + 0.5 * torch.randn_like(x1)

    y1 = model(x1, t, cond, mask)
    y2 = model(x2, t, cond, mask)

    assert not torch.allclose(y1, y2)


def test_visible_precip_is_reimposed_during_sampling():
    model = _small_model()
    cond = _cond(batch=1)
    visible = torch.randn(1, 3, 16, 16)
    mask = torch.ones(1, 16)
    mask[:, :4] = 0.0
    sample = model.sample_precip(cond, mask, precip_visible=visible, sampling_timesteps=2)
    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_ddpm_sampling_reimposes_visible_precip():
    model = _small_model()
    cond = _cond(batch=1)
    visible = torch.randn(1, 3, 16, 16)
    mask = torch.ones(1, 16)
    mask[:, :4] = 0.0
    sample = model.sample_precip(cond, mask, precip_visible=visible, sampler="ddpm")
    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    assert sample.shape == visible.shape
    assert torch.isfinite(sample).all()
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_repaint_sampling_reimposes_visible_precip_and_returns_trajectory():
    model = _small_model()
    cond = _cond(batch=1)
    visible = torch.randn(1, 3, 16, 16)
    mask = torch.ones(1, 16)
    mask[:, :4] = 0.0
    sample = model.sample_precip(
        cond,
        mask,
        precip_visible=visible,
        sampler="repaint",
        sampling_timesteps=8,
        repaint_jump_length=2,
        repaint_jump_n_sample=2,
        return_all_timesteps=True,
    )
    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    final = sample[:, -1]
    assert sample.ndim == 5
    assert sample.shape[0] == 1
    assert sample.shape[2:] == visible.shape[1:]
    assert torch.isfinite(sample).all()
    assert torch.allclose(final * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_rollout_condition_builder_uses_stacked_forcing():
    batch = {
        "background_a": torch.randn(1, 1, 16, 16),
        "reflectivity_a": torch.randn(1, 1, 16, 16),
        "precip": torch.randn(1, 3, 16, 16),
        "forcing": torch.randn(1, 2, 16, 16),
    }
    cond = _build_condition_dict(
        batch,
        background_vars=["background_a"],
        reflectivity_vars=["reflectivity_a"],
        surface_vars=[],
        forcing_vars=["f0", "f1"],
        target_size=16,
    )
    assert cond["background"].shape == (1, 1, 16, 16)
    assert cond["reflectivity"].shape == (1, 1, 16, 16)
    assert cond["surface"].shape == (1, 0, 16, 16)
    assert cond["forcing"].shape == (1, 2, 16, 16)


def test_rollout_grouped_mask_uses_variable_group_axis():
    model = _small_grouped_model()
    mask = _sample_mask(
        model,
        {"precip_mask_ratio": 0.5, "precip_mask_mode": "channel_patch"},
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 3, 16)


def test_training_mixed_height_mask_can_select_height_mask():
    model = _small_grouped_model()
    trainer = TrainerDiffMAE(model, rank=0)
    mask = trainer._sample_mask(
        batch_size=2,
        trainer_conf={
            "precip_mask_ratio": 0.5,
            "precip_mask_mode": "mixed_height",
            "mixed_height_spatial_probability": 0.0,
            "mixed_height_channel_probability": 0.0,
            "mixed_height_height_probability": 1.0,
            "height_visible_levels": [0],
        },
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 3, 2, 16)
    assert mask[:, :, 0].sum().item() == 0.0
    assert mask[:, :, 1].sum().item() > 0.0


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


def test_grouped_precip_tokenization_supports_group_specific_masks():
    model = _small_grouped_model()
    cond = _cond_grouped(batch=1)
    precip = torch.randn(1, 6, 16, 16)
    visible = torch.randn_like(precip)
    mask = torch.zeros(1, 3, 16)
    mask[:, 1, :4] = 1.0

    assert model.grouped_precip
    assert model.precip_group_index_for_channel(0) == 0
    assert model.precip_group_index_for_channel(2) == 1
    assert model.precip_group_index_for_channel(5) == 2

    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    assert pixel_mask.shape == precip.shape
    assert pixel_mask[:, 0:2].sum().item() == 0.0
    assert pixel_mask[:, 2:4, :4, :].sum().item() > 0.0
    assert pixel_mask[:, 4:6].sum().item() == 0.0

    losses = model.p_losses(precip, cond, mask)
    assert losses["model_out"].shape == precip.shape
    assert torch.isfinite(losses["loss"])

    sample = model.sample_precip(cond, mask, precip_visible=visible, sampling_timesteps=2)
    assert sample.shape == precip.shape
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_grouped_cross_self_decoder_supports_group_specific_masks():
    model = _small_grouped_model(decoder_type="cross_self")
    cond = _cond_grouped(batch=1)
    precip = torch.randn(1, 6, 16, 16)
    mask = torch.zeros(1, 3, 16)
    mask[:, 1, :4] = 1.0

    losses = model.p_losses(precip, cond, mask)
    assert losses["model_out"].shape == precip.shape
    assert torch.isfinite(losses["loss"])


def test_grouped_height_mask_supports_level_specific_visible_precip():
    model = _small_grouped_model(decoder_type="cross_self", target_attention_window_size=2)
    cond = _cond_grouped(batch=1)
    precip = torch.randn(1, 6, 16, 16)
    visible = torch.randn_like(precip)
    mask = torch.zeros(1, 3, 2, 16)
    mask[:, :, 1, :4] = 1.0

    pixel_mask = model.expand_patch_mask(mask, 16, 16)
    token_mask = model.token_mask_from_precip_mask(mask)
    assert pixel_mask.shape == precip.shape
    assert token_mask.shape == (1, 16)
    assert pixel_mask[:, 0::2].sum().item() == 0.0
    assert pixel_mask[:, 1::2, :4, :].sum().item() > 0.0

    losses = model.p_losses(precip, cond, mask)
    assert losses["model_out"].shape == precip.shape
    assert torch.isfinite(losses["loss"])

    sample = model.sample_precip(cond, mask, precip_visible=visible, sampling_timesteps=2)
    assert torch.allclose(sample * (1.0 - pixel_mask), visible * (1.0 - pixel_mask), atol=1.0e-6)


def test_anti_patch_refiner_is_initially_identity_and_checkpoint_compatible():
    base_model = _small_grouped_model(decoder_type="cross_self")
    refined_model = _small_grouped_model(
        decoder_type="cross_self",
        anti_patch_refiner={
            "enabled": True,
            "hidden_channels": 8,
            "depth": 2,
            "kernel_size": 3,
        },
    )
    load_msg = refined_model.load_state_dict(base_model.state_dict(), strict=False)
    assert load_msg.unexpected_keys == []
    assert any(key.startswith("anti_patch_refiner") for key in load_msg.missing_keys)

    cond = _cond_grouped(batch=1)
    precip = torch.randn(1, 6, 16, 16)
    t = torch.tensor([5])
    mask = torch.zeros(1, 3, 16)
    base_model.eval()
    refined_model.eval()
    with torch.no_grad():
        base_out = base_model(precip, t, cond, mask)
        refined_out = refined_model(precip, t, cond, mask)
    assert torch.allclose(refined_out, base_out, atol=1.0e-6)


def test_grouped_joint_cross_self_decoder_supports_group_specific_masks():
    model = _small_grouped_model(decoder_type="cross_self", grouped_decoder_scope="joint")
    cond = _cond_grouped(batch=1)
    precip = torch.randn(1, 6, 16, 16)
    mask = torch.zeros(1, 3, 16)
    mask[:, 1, :4] = 1.0

    losses = model.p_losses(precip, cond, mask)
    assert losses["model_out"].shape == precip.shape
    assert torch.isfinite(losses["loss"])


def test_denoise_snapshot_can_save_tensor_without_figure(tmp_path):
    model = _small_model()
    trainer = TrainerDiffMAE(model, rank=0)
    batch = {
        "background": torch.randn(1, 2, 16, 16),
        "surface": torch.randn(1, 1, 16, 16),
        "forcing": torch.randn(1, 2, 16, 16),
        "reflectivity": torch.randn(1, 1, 16, 16),
        "precip": torch.randn(1, 3, 16, 16),
    }
    mask = model.random_precip_mask(1, 1.0, torch.device("cpu"))
    losses = model.p_losses(batch["precip"], trainer._condition_dict(batch), mask)
    losses["precip_mask"] = mask
    conf = {
        "save_loc": str(tmp_path),
        "trainer": {
            "denoise_snapshot": {
                "enabled": True,
                "every_steps": 1,
                "dir": "snapshots",
                "save_snapshot": True,
                "save_figure": False,
            }
        },
    }

    trainer._maybe_save_denoise_snapshot(0, 0, conf, batch, losses)

    assert (tmp_path / "snapshots" / "epoch0000_step000000.pt").exists()
    assert not (tmp_path / "snapshots" / "epoch0000_step000000.png").exists()


def test_denoise_snapshot_can_save_figure_without_tensor(tmp_path):
    model = _small_model()
    trainer = TrainerDiffMAE(model, rank=0)
    batch = {
        "background": torch.randn(1, 2, 16, 16),
        "surface": torch.randn(1, 1, 16, 16),
        "forcing": torch.randn(1, 2, 16, 16),
        "reflectivity": torch.randn(1, 1, 16, 16),
        "precip": torch.randn(1, 3, 16, 16),
    }
    mask = model.random_precip_mask(1, 1.0, torch.device("cpu"))
    losses = model.p_losses(batch["precip"], trainer._condition_dict(batch), mask)
    losses["precip_mask"] = mask
    conf = {
        "save_loc": str(tmp_path),
        "trainer": {
            "denoise_snapshot": {
                "enabled": True,
                "every_steps": 1,
                "dir": "snapshots",
                "save_snapshot": False,
                "save_figure": True,
            }
        },
    }

    trainer._maybe_save_denoise_snapshot(0, 0, conf, batch, losses)

    assert not (tmp_path / "snapshots" / "epoch0000_step000000.pt").exists()
    assert (tmp_path / "snapshots" / "epoch0000_step000000.png").exists()


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


def test_grouped_mask_bundle_round_trip_and_runtime_conversion(tmp_path):
    bundle = build_grouped_patch_masks(
        n_times=3,
        n_groups=2,
        token_h=2,
        token_w=2,
        mask_ratio=0.5,
        mask_mode="channel_patch",
        seed=123,
    )
    out_path = tmp_path / "mask.npz"
    save_mask_bundle(
        out_path,
        patch_mask_grouped=bundle["patch_mask_grouped"],
        mask_mode=bundle["mask_mode"],
        requested_mask_ratio=bundle["requested_mask_ratio"],
        actual_group_mask_fraction=bundle["actual_group_mask_fraction"],
        group_names=["rain", "hail"],
        group_channels=[2, 1],
        image_size=(8, 8),
        patch_size=4,
        seed=123,
    )
    loaded = load_mask_bundle(out_path)
    grouped_mask, mode, requested_ratio = grouped_patch_mask_time_slice(loaded, 1)
    runtime_mask = _grouped_mask_to_runtime_mask(grouped_mask, [2, 1], mode)
    pixel_mask = _grouped_patch_to_pixel_mask(grouped_mask, [2, 1], patch_size=4, height=8, width=8)

    assert loaded["patch_mask_grouped"].shape == (3, 2, 2, 2)
    assert grouped_mask.shape == (2, 2, 2)
    assert mode == "channel_patch"
    assert requested_ratio == 0.5
    assert runtime_mask.shape == (3, 2, 2)
    assert pixel_mask.shape == (3, 8, 8)
    assert np.allclose(pixel_mask[0], pixel_mask[1])
    assert not np.allclose(pixel_mask[1], pixel_mask[2])


def test_height_mask_bundle_round_trip_and_runtime_conversion(tmp_path):
    bundle = build_grouped_patch_masks(
        n_times=2,
        n_groups=2,
        token_h=2,
        token_w=2,
        mask_ratio=1.0,
        mask_mode="height_patch",
        seed=123,
        group_channels=[2, 2],
        height_visible_levels=[0],
    )
    out_path = tmp_path / "height_mask.npz"
    save_mask_bundle(
        out_path,
        patch_mask_grouped=bundle["patch_mask_grouped"],
        mask_mode=bundle["mask_mode"],
        requested_mask_ratio=bundle["requested_mask_ratio"],
        actual_group_mask_fraction=bundle["actual_group_mask_fraction"],
        group_names=["rain", "hail"],
        group_channels=[2, 2],
        image_size=(8, 8),
        patch_size=4,
        seed=123,
    )
    loaded = load_mask_bundle(out_path)
    grouped_mask, mode, requested_ratio = grouped_patch_mask_time_slice(loaded, 0)
    runtime_mask = _grouped_mask_to_runtime_mask(grouped_mask, [2, 2], mode)
    pixel_mask = _grouped_patch_to_pixel_mask(grouped_mask, [2, 2], patch_size=4, height=8, width=8)

    assert loaded["patch_mask_grouped"].shape == (2, 2, 2, 2, 2)
    assert grouped_mask.shape == (2, 2, 2, 2)
    assert mode == "height_patch"
    assert requested_ratio == 1.0
    assert runtime_mask.shape == (2, 2, 4)
    assert pixel_mask.shape == (4, 8, 8)
    assert pixel_mask[0].sum() == 0.0
    assert pixel_mask[1].sum() == 64.0
    assert pixel_mask[2].sum() == 0.0
    assert pixel_mask[3].sum() == 64.0


def test_mixed_height_mask_bundle_can_store_spatial_channel_and_height_modes():
    bundle = build_grouped_patch_masks(
        n_times=6,
        n_groups=2,
        token_h=2,
        token_w=2,
        mask_ratio=0.5,
        mask_mode="mixed_height",
        seed=4,
        group_channels=[2, 2],
        height_visible_levels=[0],
    )

    assert bundle["patch_mask_grouped"].shape == (6, 2, 2, 2, 2)
    assert set(bundle["mask_mode"].tolist()) == {"spatial_patch", "channel_patch", "height_patch"}

    for grouped_mask, mode in zip(bundle["patch_mask_grouped"], bundle["mask_mode"]):
        runtime_mask = _grouped_mask_to_runtime_mask(grouped_mask, [2, 2], str(mode))
        if mode == "height_patch":
            assert runtime_mask.shape == (2, 2, 4)
            assert runtime_mask[:, 0].sum() == 0.0
        elif mode == "channel_patch":
            assert runtime_mask.shape == (2, 2, 4)
            assert np.allclose(runtime_mask[:, 0], runtime_mask[:, 1])
        else:
            assert runtime_mask.shape == (2, 2, 4)
            assert np.allclose(runtime_mask[0], runtime_mask[1])
            assert np.allclose(runtime_mask[:, 0], runtime_mask[:, 1])


def test_spatial_mask_runtime_conversion_returns_2d_token_mask():
    grouped_mask = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    runtime_mask = _grouped_mask_to_runtime_mask(grouped_mask, [2, 1], "spatial_patch")
    assert runtime_mask.shape == (4,)
    assert np.allclose(runtime_mask, grouped_mask[0].reshape(-1))


def test_masked_metrics_only_use_masked_entries():
    pred = np.array([[[1.0, 5.0], [2.0, 3.0]]], dtype=np.float32)
    target = np.array([[[0.0, 100.0], [2.0, 1.0]]], dtype=np.float32)
    mask = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)

    norm_metrics = compute_masked_normalized_metrics(pred, target, mask)
    phys_metrics = compute_masked_physical_metrics(pred, target, mask)

    expected_mse = ((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2.0
    expected_mae = (abs(1.0 - 0.0) + abs(3.0 - 1.0)) / 2.0
    assert np.isclose(norm_metrics["mse"], expected_mse)
    assert np.isclose(norm_metrics["mae"], expected_mae)
    assert np.isclose(phys_metrics["mse"], expected_mse)
    assert np.isclose(phys_metrics["mae"], expected_mae)
    assert np.isfinite(norm_metrics["ssim"])
