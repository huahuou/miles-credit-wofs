import torch

from credit.models.wofs_maskdit import WoFSMaskDiT


def _small_maskdit_model(
    objective="pred_v",
    grouped=False,
    maskdit_condition_token_mode="append",
):
    kwargs = dict(
        modality_channels={
            "background": 2,
            "precip": 6 if grouped else 3,
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
        maskdit_condition_token_mode=maskdit_condition_token_mode,
        maskdit_shifted_window=maskdit_condition_token_mode == "pooled_only",
        maskdit_window_shift_size=1,
        target_attention_window_size=2 if maskdit_condition_token_mode == "pooled_only" else 0,
        diffusion={
            "timesteps": 32,
            "sampling_timesteps": 4,
            "objective": objective,
            "beta_schedule": "linear",
            "ddim_sampling_eta": 0.0,
        },
    )
    if grouped:
        kwargs.update(
            precip_grouping="grouped",
            grouped_decoder_scope="per_group",
            precip_group_names=["rain", "hail", "snow"],
            precip_group_channels=[2, 2, 2],
        )
    return WoFSMaskDiT(**kwargs)


def _cond(batch=2):
    return {
        "background": torch.randn(batch, 2, 16, 16),
        "surface": torch.randn(batch, 1, 16, 16),
        "forcing": torch.randn(batch, 2, 16, 16),
        "reflectivity": torch.randn(batch, 1, 16, 16),
    }


def test_maskdit_forward_loss_and_sampling_shapes_are_finite():
    model = _small_maskdit_model()
    precip = torch.randn(2, 3, 16, 16)
    cond = _cond()
    mask = model.random_precip_mask(2, 0.75, precip.device)
    losses = model.p_losses(precip, cond, mask)
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    sample = model.sample_precip(cond, mask, sampling_timesteps=2)
    assert sample.shape == precip.shape
    assert torch.isfinite(sample).all()


def test_maskdit_grouped_height_mask_loss_is_finite():
    model = _small_maskdit_model(grouped=True, maskdit_condition_token_mode="pooled_only")
    precip = torch.randn(2, 6, 16, 16)
    cond = _cond()
    mask = model.random_height_precip_mask(2, 0.5, precip.device, visible_levels=[0])
    losses = model.p_losses(precip, cond, mask)
    assert mask.shape == (2, 3, 2, 16)
    assert losses["model_out"].shape == precip.shape
    assert torch.isfinite(losses["loss"])
