"""Smoke-test for CrossFormerWRFEnsemble: import, registry, instantiation and forward pass."""
import torch
from credit.models.wxformer.crossformer_wrf_ensemble import CrossFormerWRFEnsemble
from credit.models import model_types

# --- 1. Import and registry --------------------------------------------------
print("Import OK")
print("crossformer_wrf_ensemble in registry:", "crossformer_wrf_ensemble" in model_types)

# --- 2. Instantiate model (tiny config, CPU) ----------------------------------
model = CrossFormerWRFEnsemble(
    varname_upper_air=["T", "QVAPOR"],
    varname_surface=["T2", "Q2"],
    varname_dyn_forcing=["cos_latitude", "sin_latitude"],
    varname_forcing=[],
    varname_static=[],
    varname_diagnostic=["UP_HELI_MAX"],
    varname_boundary_upper=["T", "QVAPOR"],
    varname_boundary_surface=["T2", "Q2"],
    levels=4,
    boundary_levels=4,
    frames=1,
    embed_dim=32,
    boundary_embed_dim=16,
    time_encode_dim=12,
    dim=(16, 32, 64, 128),
    depth=(1, 1, 1, 1),
    dim_head=8,
    global_window_size=(4, 2, 1, 1),
    local_window_size=4,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    use_spectral_norm=False,
    noise_injection={"activate": True, "latent_dim": 16, "noise_scales": [0.3, 0.15, 0.05, 0.01], "encoder_noise": True},
    padding_conf={"activate": False},
)
print("Model instantiated OK")
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# --- 3. Compute expected channel counts --------------------------------------
# interior: 2 upper-air * 4 levels + 2 surface + 2 dyn_forcing = 12 channels
# boundary: 2 upper-air * 4 levels + 2 surface = 10 channels
# output:   2 upper-air * 4 levels + 2 surface + 1 diagnostic = 11 channels

B, H, W = 2, 64, 64   # H/W must be divisible by local_window * 2^4 = 64
x          = torch.randn(B, 12, 1, H, W)
x_boundary = torch.randn(B, 10, 1, H, W)
x_time     = torch.randn(B, 12)

# --- 4. Forward pass (deterministic, no noise) --------------------------------
model.eval()
with torch.no_grad():
    out = model(x, x_boundary, x_time, forecast_step=1, ensemble_size=1)
print(f"Forward (no noise) output shape: {tuple(out.shape)}")
assert out.shape == (B, 11, 1, H, W), f"Expected ({B},11,1,{H},{W}), got {tuple(out.shape)}"

# --- 5. Forward pass (ensemble noise active) ----------------------------------
with torch.no_grad():
    out_ens = model(x, x_boundary, x_time, forecast_step=1, ensemble_size=4)
print(f"Forward (with noise) output shape: {tuple(out_ens.shape)}")
assert out_ens.shape == (B, 11, 1, H, W)

# --- 6. Non-power-of-2 resolution (dynamic padding) --------------------------
H2, W2 = 70, 75   # not divisible by 64; padding should kick in
x2          = torch.randn(B, 12, 1, H2, W2)
x_boundary2 = torch.randn(B, 10, 1, H2, W2)
with torch.no_grad():
    out2 = model(x2, x_boundary2, x_time, forecast_step=1, ensemble_size=1)
print(f"Dynamic-padding output shape: {tuple(out2.shape)}")
assert out2.shape == (B, 11, 1, H2, W2), f"Expected ({B},11,1,{H2},{W2}), got {tuple(out2.shape)}"

# --- 7. add_variable API ------------------------------------------------------
prev_len = len(model.interior_embedder.slice_spec)
model.interior_embedder.add_variable("W", 4, 999)   # dummy start idx
assert len(model.interior_embedder.slice_spec) == prev_len + 1
print("add_variable API OK")

print("\nAll tests passed.")
