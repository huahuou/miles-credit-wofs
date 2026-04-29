# Aurora × CrossFormerWRFEnsemble — Architecture Analysis & Hybrid Design Guide

**Purpose**: Diagnose the weaknesses in the current `CrossFormerWRFEnsemble` input/output
projection relative to Aurora, and provide concrete implementation instructions for a
hybrid model that keeps CrossFormer's spatial processing strengths (noise injection,
anti-checkerboard UNet, local/global attention) while adopting Aurora's superior
variable-handling front-end.

---

## Table of Contents

1. [Side-by-Side Architecture Comparison](#1-side-by-side-architecture-comparison)
2. [Aurora's Key Encoder Advantages](#2-auroras-key-encoder-advantages)
   - 2.1 [Per-variable independent Conv3d embedding with history](#21-per-variable-independent-conv3d-embedding-with-history)
   - 2.2 [Explicit vertical level encoding + Perceiver aggregation](#22-explicit-vertical-level-encoding--perceiver-aggregation)
   - 2.3 [Fourier positional & temporal encoding](#23-fourier-positional--temporal-encoding)
   - 2.4 [Separate surface / upper-air token paths](#24-separate-surface--upper-air-token-paths)
3. [Aurora's Key Decoder Advantages](#3-auroras-key-decoder-advantages)
4. [What CrossFormer Does Better](#4-what-crossformer-does-better)
5. [Weakness Analysis of VariableTokenEmbedder (Summation Design)](#5-weakness-analysis-of-variabletokenembedder-summation-design)
6. [Variable-Flexibility Trade-off: Both Models](#6-variable-flexibility-trade-off-both-models)
7. [Proposed Hybrid: AuroraCrossFormerWRF](#7-proposed-hybrid-auroracrossformerwrf)
   - 7.1 [Architecture data-flow](#71-architecture-data-flow)
   - 7.2 [Component 1 — VerticalLevelEmbedder (replaces VariableTokenEmbedder)](#72-component-1--verticallevelsembedder-replaces-variabletokenembedder)
   - 7.3 [Component 2 — Fourier lat/lon/level/time positional encoding](#73-component-2--fourier-latlonleveltime-positional-encoding)
   - 7.4 [Component 3 — Enhanced boundary conditioning](#74-component-3--enhanced-boundary-conditioning)
   - 7.5 [Component 4 — Keep CrossFormer backbone + noise + decoder unchanged](#75-component-4--keep-crossformer-backbone--noise--decoder-unchanged)
8. [Implementation: Step-by-Step Code Guide](#8-implementation-step-by-step-code-guide)
   - 8.1 [VerticalLevelEmbedder](#81-verticallevelsembedder)
   - 8.2 [Fourier Encoding for LAM](#82-fourier-encoding-for-lam)
   - 8.3 [Wiring into CrossFormerWRFEnsemble.__init__](#83-wiring-into-crossformerwrfensembleinit)
   - 8.4 [Wiring into CrossFormerWRFEnsemble.forward](#84-wiring-into-crossformerwrfensembleforward)
9. [Config Changes](#9-config-changes)
10. [Variable Flexibility After the Change](#10-variable-flexibility-after-the-change)
11. [Memory & Compute Impact](#11-memory--compute-impact)
12. [Training Strategy](#12-training-strategy)

---

## 1. Side-by-Side Architecture Comparison

| Concern | Aurora (Perceiver3DEncoder/Decoder) | CrossFormerWRFEnsemble (current) |
|---|---|---|
| **Variable input projection** | Per-variable `Conv3d(1, D, kernel=(T,P,P))` in `LevelPatchEmbed`; weights summed with shared bias; truly independent per variable | Per-variable `Conv2d(n_ch, D, 1×1)` in `VariableTokenEmbedder`; outputs **summed** with scalar sigmoid gate |
| **Vertical level handling** | Separate per-level patch tokens; explicit `levels_expansion` Fourier encode; Perceiver aggregation across L levels | All L levels concatenated as flat channels; single Conv2d projects the whole block together |
| **History / multi-frame** | `Conv3d` temporal kernel `(T, P, P)`; history-aware embedding without channel concatenation | Frames reshaped into channels `(C*T, H, W)` before Conv2d; no temporal ordering awareness |
| **Surface vs upper-air** | Fully separate token streams; surface has dedicated `surf_level_encoding` learned bias | No separation; all variables treated identically in `VariableTokenEmbedder` |
| **Positional encoding** | Fourier expansion of lat/lon (patch centre), patch area (scale), lead time, absolute time | `cos_lat / sin_lat / cos_lon / sin_lon` passed as **input channels** (not additive encodings) |
| **Boundary / context conditioning** | N/A (global model); handled implicitly via full-attention backbone | Global-average-pool of boundary emb → FiLM (γ,β) on interior emb — loses spatial structure |
| **Backbone** | 3-D Swin Transformer (ViT-style tokens, no UNet) | CrossFormer 4-stage UNet with CrossEmbedLayer + local/global window attention |
| **Decoder** | Per-variable linear heads → `unpatchify`; Perceiver deaggregation for vertical levels | Anti-checkerboard `AnticheckboardUpBlock` UNet decoder + `VariableTokenDecoder` |
| **Noise / stochasticity** | None | `StochasticDecompositionLayer` at 4 decoder scales + optional encoder scales |
| **Boundary padding** | N/A | `TensorPadding` (mirror mode); physically correct for LAM |
| **Variable set at inference** | Yes — each variable has its own weight key; call with any subset | Yes — exclude channels from slice_spec; projector exists but is not called |
| **Adding new variable without backbone retraining** | Yes — new `nn.Parameter` weight tensor; load with `strict=False` | Yes — `add_variable()` zero-inits a new Conv2d; load with `strict=False` |
| **# parameters (embedding front-end, WoFS default)** | ~2–3 M (Conv3d per-var weights + Perceiver aggregator) | ~0.5–1 M (flat Conv2d per-var, no vertical structure) |

---

## 2. Aurora's Key Encoder Advantages

### 2.1 Per-variable independent Conv3d embedding with history

Aurora's `LevelPatchEmbed` embeds each variable independently using a `Conv3d` with a
**temporal kernel** that spans the full history window:

```python
# Shape: (C_out=D, C_in=1, T=history_size, H=patch_size, W=patch_size)
self.weights[name] = nn.Parameter(torch.empty(embed_dim, 1, T, patch_size, patch_size))
```

When called with `T < history_size`, it automatically truncates the kernel in the temporal
dimension. All variable weights are **concatenated** along the input-channel dimension into
a single Conv3d call, so there is no Python loop overhead at forward time:

```python
weight = torch.cat([self.weights[name][:, :, :T] for name in var_names], dim=1)
proj = F.conv3d(x, weight, self.bias, stride=(T, P, P))  # (B, D, 1, H/P, W/P)
```

**Why this is better than CrossFormer's flat Conv2d summation:**

- Levels of the same variable (e.g. T at 1000 hPa and T at 500 hPa) are embedded through
  their own weight tensor, not merged before projection. The network can learn that T at
  surface is physically different from T at the tropopause without any coupling through a
  shared convolutional weight.
- History frames are embedded with a temporal convolution whose receptive field naturally
  encodes the temporal trend. CrossFormer's reshape `(C*T, H, W)` treats frame-2 as if it
  were an unrelated variable at a different channel index; the network cannot generalize
  across different values of T without seeing all combinations.
- The summation gate in CrossFormer is a **scalar** per variable. All spatial positions get
  the same gate value. Aurora achieves spatial specialization implicitly because each
  variable's Conv3d weight can learn spatially different patterns.

### 2.2 Explicit vertical level encoding + Perceiver aggregation

After patch embedding, Aurora adds a **Fourier vertical level encoding** to every atmospheric
token:

```python
atmos_levels_encode = levels_expansion(atmos_levels_tensor, embed_dim)  # (C_A, D)
atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)        # (C_A, D)
x_atmos = x_atmos + atmos_levels_embed[None, :, None, :]
```

This lets the backbone distinguish eta level 1 from eta level 17 without relying on
positional index in the flattened channel vector.

The **Perceiver aggregation** then compresses L physical levels → C latent levels using
cross-attention with learned latent queries:

```python
# latent queries: (C_latent - 1, D)  learnable
x_atmos = level_agg(latents, x_atmos)  # cross-attention: (B*L, C_latent, D)
```

**For WoFS with 17 eta levels**, this is significant: the model learns which physical
levels are most predictively relevant for the next time step, rather than treating the
17 levels as 17 independent channels with equal weight.

### 2.3 Fourier positional & temporal encoding

Aurora uses a `FourierExpansion` module that produces log-spaced sin/cos encodings:

| Encoding | Input | Range | Physical meaning |
|---|---|---|---|
| `pos_expansion(lat, lon)` | patch centre lat/lon in radians | [min_area, π] | spatial position on the sphere |
| `scale_expansion(lat, lon, patch_size)` | patch area | [min_area, max_area] | local grid spacing |
| `lead_time_expansion(hours)` | 0–120 h | [1 h, 120 h] | forecast lead time |
| `absolute_time_expansion(unix_hours)` | wall-clock time | [1 h, 1 year] | diurnal/seasonal cycle |
| `levels_expansion(eta)` | 0–1 (normalised pressure) | [0.01, 1] | vertical position |

These encodings are **additive** to the token embedding — they do not consume input
channels. CrossFormer currently uses `cos_lat / sin_lat / cos_lon / sin_lon` as **input
variables** in `varname_dyn_forcing`, which:
- Wastes model capacity — the projection has to "learn" that these channels encode position
  rather than receiving a proper positional encoding that the architecture already knows how
  to use.
- Couples positional information to the variable gate logic in `VariableTokenEmbedder`.

### 2.4 Separate surface / upper-air token paths

Aurora maintains fully separate streams for surface-level and atmospheric variables until
after vertical aggregation. Surface tokens receive a dedicated `surf_level_encoding` learnable
bias that teaches the backbone "this token is from the 2-D surface, not from a pressure
level." This distinction matters enormously for WoFS: `T2` (2-m temperature) and `T` at
eta level 1 (lowest model level) are physically related but distinct; conflating them in a
flat channel vector is an approximation.

---

## 3. Aurora's Key Decoder Advantages

Aurora's `Perceiver3DDecoder` decodes each variable through its own linear head:

```python
self.surf_heads  = {name: Linear(D, patch_size**2) for name in surf_vars}
self.atmos_heads = {name: Linear(D, patch_size**2) for name in atmos_vars}
```

The Perceiver **de-aggregates** latent levels back to physical levels using a second
PerceiverResampler with learned level queries. This is the decoder mirror of the encoder
aggregation and ensures that the 17 output eta levels are reconstructed from a compact
learned representation rather than being a simple linear projection.

For the WoFS task, the current `VariableTokenDecoder` (per-variable Conv2d) is actually
**adequate** — it is structurally equivalent to Aurora's per-variable linear head after
the feature map has been upsampled to full resolution. The main decoder gap in CrossFormer
is not the output projection head but rather the **absence of vertical de-aggregation**.

---

## 4. What CrossFormer Does Better

These components are **superior to Aurora** for the WoFS emulator task and must be
preserved in any hybrid:

| Component | Why CrossFormer wins |
|---|---|
| **Anti-checkerboard UNet decoder** | Dense spatial prediction at full 3-km resolution. Aurora's ViT decoder uses patch-then-unpatchify which blurs sub-patch detail. The 4-stage convolutional upsampler preserves fine convective-scale structure. |
| **StochasticDecompositionLayer noise injection** | Multi-scale ensemble generation with physically motivated scale-dependent noise. Aurora has no mechanism for stochastic inference. This is the most critical differentiator for WoFS probabilistic forecasting. |
| **Local + global CrossFormer attention** | The alternating local (24 km) and global attention in the UNet encoder is well-matched to WoFS mesoscale dynamics. Aurora's Swin3D operates on 3-D patches and would require significant modification for 2-D LAM. |
| **TensorPadding (mirror mode)** | Physically correct reflective boundary for the WoFS LAM domain. No analog in Aurora. |
| **FiLM boundary conditioning** | Explicit conditioning on lateral boundary conditions from the parent model. Aurora has no concept of boundary nudging. |
| **KCRPS ensemble loss + two-phase training** | Well-matched two-phase strategy for deterministic then ensemble training. |

---

## 5. Weakness Analysis of VariableTokenEmbedder (Summation Design)

The current `VariableTokenEmbedder` has three specific weaknesses that Aurora's design
solves:

### 5.1 Vertical level collapse

```
VariableTokenEmbedder for 'T':
  x[:, 0:17, :, :]  →  Conv2d(17, D, 1×1)  →  gate_T × out
```

The single `Conv2d(17, D)` must learn a 17→D projection that captures the entire vertical
profile of T in one weight matrix. The backbone never sees "T at level 7" as a separate
entity — the vertical structure is entirely encoded in the 17 rows of this weight matrix.
If you add or remove levels you must rebuild the entire weight matrix. Aurora's approach
gives each level its own 1-channel weight, so a new level is just a new row appended with
zero-init.

### 5.2 No temporal structure

```
CrossFormer flattens T=2 frames: C*T channels as input to Conv2d(C*T, D, 1×1)
Aurora uses: Conv3d(1, D, kernel=(T, P, P)) per variable
```

With C=112 channels and T=2, CrossFormer feeds a 224-channel flat vector into the Conv2d.
The kernel cannot distinguish "T at level 3, frame 0" from "QVAPOR at level 3, frame 1"
except by their absolute channel index. Aurora's Conv3d temporal kernel explicitly treats
the time axis as a structured dimension separate from the variable axis.

### 5.3 Scalar gate cannot model spatial inactivation

The sigmoid gate `torch.sigmoid(gate_logits[name])` is a single scalar. It modulates the
entire variable uniformly across all spatial positions. Real atmospheric data often has
spatially localised missing data, instrument dropouts, or regime changes where a variable
should contribute more in some regions (e.g., COMPOSITE_REFL_10CM is near-zero outside
active convection). A spatially-varying gating (e.g., a lightweight `Conv2d(D, D, 1)` gate
or an attention weight over spatial positions) would capture this.

---

## 6. Variable-Flexibility Trade-off: Both Models

A common concern is whether adopting a per-variable-per-level embedding sacrifices the
"add a variable without retraining the backbone" property.

**Aurora**: Each variable is a separate `nn.Parameter` key in `LevelPatchEmbed.weights`.
Adding a variable = adding one new `(D, 1, T, P, P)` parameter tensor. The backbone is
unchanged. Load with `strict=False`.  **Variable set can change at inference** by passing a
different `var_names` list to `forward()`.

**CrossFormer (current)**: `add_variable()` adds a new Conv2d leaf and zero-inits it.
The backbone is unchanged. Load with `strict=False`. Variable set can change at inference
by adjusting the slice_spec.

**Proposed hybrid VerticalLevelEmbedder**: Each (variable, level) pair becomes a separate
learnable weight tensor of shape `(D, 1, T, 1, 1)` (a 1×1 spatial kernel, temporal depth T).
This is strictly more flexible than the current Conv2d(17, D) approach:
- Add a new level: append one new weight tensor, zero-init. Backbone unchanged.
- Add a new variable: append L new weight tensors (one per level). Backbone unchanged.
- Remove a variable at inference: skip its weight tensors in the concatenation.
- The backbone embed_dim D is unchanged regardless of how many (var, level) pairs exist.

**The flexibility property is preserved** — and actually extended to per-level granularity.

---

## 7. Proposed Hybrid: AuroraCrossFormerWRF

The target architecture replaces only the encoder and decoder **front-ends** while keeping
the CrossFormer UNet backbone and all noise/boundary machinery intact.

### 7.1 Architecture data-flow

```
Interior x   (B, C_in, T, H, W)            Boundary xb  (B, C_b, T, H, W)
      │                                              │
      ▼                                              ▼
┌──────────────────────────────────────┐  ┌────────────────────────────────────┐
│  VerticalLevelEmbedder               │  │  BoundaryLevelEmbedder             │
│  per-(var,level) Conv2d(1,D,1×1)     │  │  same design, boundary_embed_dim   │
│  + Fourier level encoding            │  └────────────────┬───────────────────┘
│  + lightweight level aggregation     │                   │ (B, bnd_D, H, W)
│  (B, embed_dim, H, W)                │                   │
└──────────────────┬───────────────────┘                   │
                   │                                        │
      ┌────────────▼────────────────────────────────────────▼───┐
      │  FiLM conditioning (enhanced):                           │
      │  xb → spatial cross-attention pool → (B, bnd_D)         │
      │  time_encode → Fourier embed → (B, T_dim)               │
      │  concat → MLP → γ, β → modulate x_emb                   │
      └──────────────────────┬───────────────────────────────────┘
                             │
      ┌──────────────────────▼──────────────────────────────────────┐
      │  Fourier lat/lon positional encoding (additive)              │
      │  x_emb = x_emb + pos_enc(lat, lon, D)                       │
      └──────────────────────┬──────────────────────────────────────┘
                             │ (B, embed_dim, H, W)
                             │  pad to multiple of ws×2^4
                             ▼
              ┌──────────────────────────────┐
              │  CrossFormer Encoder (4 stgs) │  ← unchanged
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  Anti-checkerboard Decoder    │  ← unchanged
              │  + StochasticDecomposition    │  ← unchanged
              │    noise injection            │
              └──────────────┬───────────────┘
                             │ (B, feat_dim, H, W)
                             ▼
              ┌──────────────────────────────┐
              │  VariableTokenDecoder         │  ← unchanged
              │  per-var Conv2d heads         │
              └──────────────────────────────┘
```

### 7.2 Component 1 — VerticalLevelEmbedder (replaces VariableTokenEmbedder)

The key change. For each (variable, level) pair, a separate 1×1 Conv2d with 1 input
channel is registered:

```python
class VerticalLevelEmbedder(nn.Module):
    """
    Per-(variable, level) independent 1×1 projection.

    For upper-air variable V with L levels:
        register L conv keys: 'V_0', 'V_1', ..., 'V_{L-1}'
        each: Conv2d(frames, D, 1×1)  -- frames channels per level

    For surface / forcing / static variable S:
        register 1 conv key: 'S'
        Conv2d(frames, D, 1×1)

    All projections are SUMMED (preserving backbone width = embed_dim).
    Each key has an independent learnable scalar gate (sigmoid).

    Level encoding is ADDED before summation so the network can distinguish
    level 3 from level 14 of the same variable.
    """
```

**Why `frames` input channels instead of 1?**  
For `T=1` (single frame), `Conv2d(1, D, 1)` is identical to Aurora's approach. For
`T>1` (multi-frame rollout input), the T frames of the same variable at the same level
share a single weight matrix — this is a 2-D version of Aurora's Conv3d temporal kernel
and naturally encodes the temporal difference without channel-index aliasing.

### 7.3 Component 2 — Fourier lat/lon/level/time positional encoding

Replace the `cos_lat / sin_lat / cos_lon / sin_lon` input channels with **additive** encodings
built from the same Fourier expansion used in Aurora. The critical gain: dyn_forcing
channels are freed up (or can be removed from the variable list), reducing `C_in` and
speeding up the embedder.

```python
# Fourier pos encoding for LAM: use pixel grid, not spherical coordinates
# lat: (H,)  lon: (W,)  →  pos_enc: (1, D, H, W)  additive after embedder
def lam_fourier_pos_enc(lat_1d, lon_1d, embed_dim) -> Tensor:  # (1, D, H, W)
    ...
```

Level encoding remains an additive `(1, D, 1, 1)` per-level offset applied before
the summation across levels inside VerticalLevelEmbedder.

### 7.4 Component 3 — Enhanced boundary conditioning

The current FiLM uses `xb_emb.mean(dim=(-2,-1))` which discards all spatial structure of
the boundary. An improvement is to use a multi-scale spatial descriptor:

```python
# Instead of single global pool, compute multi-scale pooling:
xb_s1 = xb_emb.mean(dim=(-2,-1))                      # (B, bnd_D)  global
# Practical improvement: use 4-quadrant pooling (retains coarse spatial structure)
xb_q  = F.avg_pool2d(xb_emb, (H//2, W//2))            # (B, bnd_D, 2, 2)
xb_q  = xb_q.flatten(1)                               # (B, 4 × bnd_D)
film_cond = cat([xb_q, xb_s1, time_enc])
```

#### Boundary input choices at runtime

Three approaches are supported, selected by a single config flag `trainer.tendency_boundary`:

| Mode | `tendency_boundary` | `x_boundary` content | Availability |
|---|---|---|---|
| **Outer-zone (original)** | `False` (default) | Frozen t=0 lateral boundary zone from dataset | Training + inference |
| **State tendency** | `True` | `y_pred_{k-1} − x_{k-1}` (zero at k=1) | Training + inference, no external data needed |
| **HRRR-driven** (future) | `False` + time-varying dataset | True time-varying boundary from HRRR | Requires HRRR archive |

**Recommended: `tendency_boundary: True`**

The state tendency is the most physically motivated signal when HRRR data is unavailable:

- At **step 1 (t=0)**: boundary = **0** — no prior tendency known, correct by construction. The model relies entirely on its interior state, which is the right thing to do.
- At **step k>1**: boundary = `y_pred_{k-1} − x_{k-1}` — how much the atmosphere moved in the previous step. With `residual_prediction: True`, this is exactly the previous model residual output.
- The FiLM-conditioned interior embedding sees "how fast things were changing last step", giving the model the velocity context it needs to predict the next residual.
- No external data, no dataset changes, no additional I/O.

**Config required when using tendency boundary:**

```yaml
trainer:
  tendency_boundary: True    # compute boundary from state tendency in-loop

model:
  # Boundary variable lists MUST match interior prognostic variables
  varname_boundary_upper:   ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT']   # = data.variables
  varname_boundary_surface: ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT']  # = data.surface_variables (NO XLAND/HGT)
  boundary_levels: 17   # = data.levels
  boundary_embed_dim: 64
```

The dataset still loads `x_boundary` from zarr (backward compatible), but the trainer
ignores it when `tendency_boundary: True` and computes the tendency in-loop instead.

### 7.5 Component 4 — Keep CrossFormer backbone + noise + decoder unchanged

No changes to:
- `encoder_layers` (CrossEmbedLayer + Transformer)
- `up_block1/2/3/4` (AnticheckboardUpBlock)
- `StochasticDecompositionLayer` noise injection
- `VariableTokenDecoder` output heads
- `TensorPadding` mirror padding
- KCRPS / two-phase training strategy

---

## 8. Implementation: Step-by-Step Code Guide

### 8.1 VerticalLevelEmbedder

Add to `crossformer_wrf_ensemble.py` (replace `VariableTokenEmbedder` class, or add as a
new class and select via config):

```python
class VerticalLevelEmbedder(nn.Module):
    """
    Per-(variable, level) independent projection with explicit level encoding.

    slice_spec entries:
        upper-air:  (name, start_ch, n_levels, frames)
        surface:    (name, start_ch, 1,        frames)

    All projections are summed into (B, embed_dim, H, W).
    """

    def __init__(
        self,
        upper_spec: List[Tuple[str, int, int, int]],   # (name, start, n_levels, frames)
        surface_spec: List[Tuple[str, int, int]],       # (name, start, frames)
        embed_dim: int,
        level_encode_dim: int = 32,                     # Fourier level encoding dim
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.upper_spec = upper_spec
        self.surface_spec = surface_spec

        # Per-(var, level) Conv2d projectors
        self.projectors = nn.ModuleDict()
        self.gate_logits = nn.ParameterDict()

        for name, start, n_levels, frames in upper_spec:
            for lev in range(n_levels):
                key = f"{name}_lev{lev}"
                self.projectors[key] = nn.Conv2d(frames, embed_dim, kernel_size=1)
                self.gate_logits[key] = nn.Parameter(torch.zeros(1))

        for name, start, frames in surface_spec:
            self.projectors[name] = nn.Conv2d(frames, embed_dim, kernel_size=1)
            self.gate_logits[name] = nn.Parameter(torch.zeros(1))

        # Level encoding: maps level index (normalised 0–1) to embed_dim via Fourier
        # Stored as a buffer; not a parameter (fixed Fourier basis)
        self.level_encode_dim = level_encode_dim
        # Learnable linear to project Fourier features to embed_dim
        self.level_proj = nn.Linear(level_encode_dim, embed_dim)

        # Channel-wise LayerNorm after summation
        self.norm = nn.LayerNorm(embed_dim)

    def _fourier_level_enc(self, lev_idx: int, n_levels: int) -> torch.Tensor:
        """Fixed sin/cos encoding for a single level index → (embed_dim,)."""
        t = lev_idx / max(n_levels - 1, 1)  # normalise to [0, 1]
        half = self.level_encode_dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        enc = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=0)
        return enc  # (level_encode_dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_total, H, W)  where C_total = sum of all channels × frames
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype
        out: Optional[torch.Tensor] = None

        # --- upper-air variables ---
        for name, start, n_levels, frames in self.upper_spec:
            for lev in range(n_levels):
                key = f"{name}_lev{lev}"
                ch_start = start + lev * frames
                x_lev = x[:, ch_start : ch_start + frames]      # (B, frames, H, W)

                # 1×1 projection
                proj = self.projectors[key](x_lev)               # (B, D, H, W)

                # Additive level encoding (broadcast over B, H, W)
                lev_enc = self._fourier_level_enc(lev, n_levels).to(device=device, dtype=dtype)
                lev_enc = self.level_proj(lev_enc)               # (D,)
                proj = proj + lev_enc[None, :, None, None]       # broadcast

                gate = torch.sigmoid(self.gate_logits[key])
                out = gate * proj if out is None else out + gate * proj

        # --- surface / forcing / static variables ---
        for name, start, frames in self.surface_spec:
            x_s = x[:, start : start + frames]
            proj = self.projectors[name](x_s)                    # (B, D, H, W)
            gate = torch.sigmoid(self.gate_logits[name])
            out = gate * proj if out is None else out + gate * proj

        if out is None:
            raise RuntimeError("VerticalLevelEmbedder: no variables configured")

        # LayerNorm over channel axis
        out = out.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        out = self.norm(out)
        return out.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

    def add_variable(
        self, name: str, n_levels: int, start_channel: int, frames: int, is_upper: bool
    ) -> None:
        """Zero-initialise projectors for a new variable (fine-tuning)."""
        if is_upper:
            for lev in range(n_levels):
                key = f"{name}_lev{lev}"
                if key not in self.projectors:
                    proj = nn.Conv2d(frames, self.embed_dim, kernel_size=1)
                    nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
                    self.projectors[key] = proj
                    self.gate_logits[key] = nn.Parameter(torch.zeros(1))
            self.upper_spec.append((name, start_channel, n_levels, frames))
        else:
            if name not in self.projectors:
                proj = nn.Conv2d(frames, self.embed_dim, kernel_size=1)
                nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
                self.projectors[name] = proj
                self.gate_logits[name] = nn.Parameter(torch.zeros(1))
            self.surface_spec.append((name, start_channel, frames))
```

### 8.2 Fourier Encoding for LAM

Add a standalone helper (can reuse Aurora's `FourierExpansion` or implement from scratch):

```python
import math
import torch
import torch.nn as nn

def lam_fourier_pos_enc(
    lat_1d: torch.Tensor,   # (H,) in degrees or normalised [0,1]
    lon_1d: torch.Tensor,   # (W,) in degrees or normalised [0,1]
    embed_dim: int,
    min_wavelength: float = 0.01,
    max_wavelength: float = 1.0,
) -> torch.Tensor:
    """
    Returns (1, embed_dim, H, W) additive positional encoding.

    Uses sin/cos at log-spaced frequencies, matching Aurora's FourierExpansion.
    Normalise lat/lon to [0, 1] before calling:
        lat_norm = (lat - lat.min()) / (lat.max() - lat.min())
    """
    half = embed_dim // 4           # each of lat/lon gets embed_dim/2 channels
    freqs = torch.logspace(
        math.log10(min_wavelength),
        math.log10(max_wavelength),
        half, base=10,
        device=lat_1d.device, dtype=lat_1d.dtype
    )
    # lat: (H,) → (H, half)  sin/cos
    lat_enc = torch.cat([
        torch.sin(2 * math.pi * lat_1d[:, None] / freqs[None, :]),
        torch.cos(2 * math.pi * lat_1d[:, None] / freqs[None, :]),
    ], dim=-1)                      # (H, embed_dim/2)

    lon_enc = torch.cat([
        torch.sin(2 * math.pi * lon_1d[:, None] / freqs[None, :]),
        torch.cos(2 * math.pi * lon_1d[:, None] / freqs[None, :]),
    ], dim=-1)                      # (W, embed_dim/2)

    # outer-product into (H, W, embed_dim) then to (1, D, H, W)
    enc = torch.cat([
        lat_enc[:, None, :].expand(-1, lon_enc.shape[0], -1),
        lon_enc[None, :, :].expand(lat_enc.shape[0], -1, -1),
    ], dim=-1)                      # (H, W, embed_dim)
    return enc.permute(2, 0, 1).unsqueeze(0)   # (1, D, H, W)
```

### 8.3 Wiring into CrossFormerWRFEnsemble.__init__

In `__init__`, change the spec-building and embedder construction:

```python
# --- NEW: build upper and surface specs separately ---
upper_spec: List[Tuple[str, int, int, int]] = []    # (name, start, n_levels, frames)
surface_spec: List[Tuple[str, int, int]] = []        # (name, start, frames)
cursor = 0

for name in varname_upper_air:
    upper_spec.append((name, cursor, levels, frames))
    cursor += levels * frames

for name in [*varname_surface, *varname_dyn_forcing, *varname_forcing, *varname_static]:
    surface_spec.append((name, cursor, frames))
    cursor += 1 * frames

# --- Replace VariableTokenEmbedder with VerticalLevelEmbedder ---
self.interior_embedder = VerticalLevelEmbedder(
    upper_spec=upper_spec,
    surface_spec=surface_spec,
    embed_dim=embed_dim,
    level_encode_dim=level_encode_dim,   # new config param, default 32
)

# --- Boundary embedder (same pattern) ---
bnd_upper_spec: List[Tuple[str, int, int, int]] = []
bnd_surface_spec: List[Tuple[str, int, int]] = []
cursor = 0
for name in varname_boundary_upper:
    bnd_upper_spec.append((f"bnd_{name}", cursor, boundary_levels, frames))
    cursor += boundary_levels * frames
for name in varname_boundary_surface:
    bnd_surface_spec.append((f"bnd_{name}", cursor, frames))
    cursor += 1 * frames

self.boundary_embedder = VerticalLevelEmbedder(
    upper_spec=bnd_upper_spec,
    surface_spec=bnd_surface_spec,
    embed_dim=boundary_embed_dim,
    level_encode_dim=level_encode_dim,
)

# --- Fourier position encoding (learnable linear projection) ---
self.pos_enc_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
# stored as a buffer once spatial size is known (lazy init in forward)
self._pos_enc_cache: Optional[torch.Tensor] = None
self._pos_enc_shape: Optional[Tuple[int, int]] = None

# --- Enhanced FiLM: 4-quadrant boundary pooling ---
# 4 quadrants × boundary_embed_dim + global_pool × boundary_embed_dim + time_encode_dim
film_in = 5 * boundary_embed_dim + time_encode_dim
self.film_mlp = nn.Sequential(
    nn.Linear(film_in, film_in * 2),
    nn.SiLU(),
    nn.Linear(film_in * 2, 2 * embed_dim),
)
```

### 8.4 Wiring into CrossFormerWRFEnsemble.forward

```python
def forward(self, x, x_boundary, x_time_encode, forecast_step=0, ensemble_size=1):
    ...
    # (existing TensorPadding + flatten T → 2D)
    x_2d = ...    # (B, C, H, W)
    xb_2d = ...   # (B, C_b, H, W)

    # --- NEW: VerticalLevelEmbedder ---
    x_emb  = self.interior_embedder(x_2d)    # (B, embed_dim, H, W)
    xb_emb = self.boundary_embedder(xb_2d)   # (B, bnd_dim,   H, W)

    # --- NEW: additive Fourier pos encoding ---
    H, W = x_emb.shape[-2:]
    if self._pos_enc_shape != (H, W):
        # lazy init — compute once per spatial size change
        lat_norm = torch.linspace(0, 1, H, device=x.device)
        lon_norm = torch.linspace(0, 1, W, device=x.device)
        pos_enc = lam_fourier_pos_enc(lat_norm, lon_norm, self.embed_dim)
        self._pos_enc_cache = pos_enc.to(device=x_emb.device, dtype=x_emb.dtype)
        self._pos_enc_shape = (H, W)
    x_emb = x_emb + self._pos_enc_cache

    # --- NEW: enhanced FiLM with 4-quadrant boundary ---
    xb_global = xb_emb.mean(dim=(-2,-1))                         # (B, bnd_D)
    xb_q = F.avg_pool2d(xb_emb, (max(H//2,1), max(W//2,1)))     # (B, bnd_D, 2, 2)
    xb_q = xb_q.flatten(1)                                       # (B, 4*bnd_D)
    film_cond = torch.cat([xb_global, xb_q, x_time_encode], dim=1)
    gamma_beta = self.film_mlp(film_cond)
    gamma, beta = gamma_beta.chunk(2, dim=1)
    x_emb = gamma.view(B, -1, 1, 1) * x_emb + beta.view(B, -1, 1, 1)

    # --- REST IS UNCHANGED: window-pad → CrossFormer encoder → decoder ---
    ...
```

---

## 9. Config Changes

Only two new keys are needed under `model`:

```yaml
model:
  # Existing keys unchanged
  varname_upper_air: [...]
  varname_surface: [...]
  varname_dyn_forcing: []            # <-- can be emptied if using Fourier pos enc
  embed_dim: 128
  ...

  # New keys
  level_encode_dim: 32               # Fourier dim for vertical level encoding
  use_vertical_level_embedder: True  # False = use old VariableTokenEmbedder (backwards compat)
  use_fourier_pos_enc: True          # False = old dyn_forcing channels for position
  film_quadrant_pool: True           # False = old global-only FiLM
```

**If `use_fourier_pos_enc: True`**: remove `cos_latitude / sin_latitude / cos_longitude /
sin_longitude` from `data.dynamic_forcing_variables` and `model.varname_dyn_forcing`.
The model is now position-aware through the encoding, not through input channels. Keep
`cos_julian_day / sin_julian_day / cos_local_time / sin_local_time / cos_solar_zenith_angle /
insolation` as input channels — these are **physical** inputs (not mere positional encodings)
and should remain in the variable list.

---

## 10. Variable Flexibility After the Change

| Scenario | Current VariableTokenEmbedder | Proposed VerticalLevelEmbedder |
|---|---|---|
| Add new 2-D surface var | `add_variable(name, 1, offset)` → zero-init Conv2d. `strict=False`. | `add_variable(name, 1, offset, frames, is_upper=False)`. Same mechanism. |
| Add new 3-D var (same L levels) | `add_variable(name, L, offset)` → one Conv2d(L, D). | `add_variable(name, L, offset, frames, is_upper=True)` → L new Conv2d(frames, D). More granular. |
| Add 3-D var with different L' levels | `add_variable(name, L', offset)` → one Conv2d(L', D). | Same: L' new 1-channel projectors. **Easier**: no weight shape change at all. |
| Remove variable at inference | Set slice_spec to exclude. Projector unused. | Same: skip its keys in forward loop. |
| Change number of vertical levels | Must rebuild Conv2d(old_L, D) → Conv2d(new_L, D). **Backbone weight must change.** | Add/remove individual per-level projectors. Backbone unchanged. |

The per-level granularity is the decisive improvement: Aurora's design handles heterogeneous
vertical coordinate systems (e.g. mixing WRF eta levels with a new ERA5-based variable at
pressure levels) without architectural changes.

---

## 11. Memory & Compute Impact

For WoFS defaults: 6 upper-air vars × 17 levels = 102 per-level projectors (each Conv2d(1,128,1))
+ 6 surface + 6 dyn_forcing = 12 surface projectors. Total: 114 projectors.

Each `Conv2d(1, 128, 1)` has 128 + 128 = 256 parameters.
Total embedder parameters: 114 × 256 ≈ **29 K parameters** (plus level_proj Linear(32,128) ≈ 4 K).

Compare to current `Conv2d(17, 128, 1)` per upper-air var: 6 × 17 × 128 = 13,056 + bias ≈ same
order. The per-level design has **more parameters** in the embedder front-end but these are
small relative to the 20 M backbone. The added cost is the forward loop over 114 projectors
instead of 12 — use `torch.cat` + a single batched `F.conv2d` call for the projectors
sharing the same `frames` depth to eliminate Python loop overhead.

Fourier pos encoding: just an addition, zero additional parameters after the projection
linear (`Conv2d(D, D, 1)` ≈ 16 K params for D=128).

Enhanced FiLM (4-quadrant): FiLM MLP grows from `film_in = bnd_D + time_D` to
`5*bnd_D + time_D`. For `bnd_D=64, time_D=12`: 332→532 input, MLP size grows by ~60%.
Negligible relative to backbone.

---

## 12. Training Strategy

No change to the two-phase training schedule. The phase-1/phase-2 boundary is still
at the `noise_injection.activate` flag. Weight transfer between a pretrained
VariableTokenEmbedder checkpoint and the new VerticalLevelEmbedder requires a
**one-time weight migration script**:

```python
# Migrate per-variable Conv2d(n_levels*frames, D) weights to per-level Conv2d(frames, D)
# old key:  "interior_embedder.projectors.T.weight"  shape (D, 17*frames, 1, 1)
# new keys: "interior_embedder.projectors.T_lev0.weight" ... "T_lev16.weight"
#            each shape (D, frames, 1, 1)
def migrate_embedder_weights(old_ckpt: dict, varname_upper_air, levels, frames) -> dict:
    new_ckpt = {k: v for k, v in old_ckpt.items()
                if "interior_embedder.projectors" not in k
                and "boundary_embedder.projectors" not in k}
    for name in varname_upper_air:
        old_w = old_ckpt[f"interior_embedder.projectors.{name}.weight"]  # (D, L*F, 1, 1)
        for lev in range(levels):
            chunk = old_w[:, lev*frames:(lev+1)*frames, :, :]  # (D, frames, 1, 1)
            new_ckpt[f"interior_embedder.projectors.{name}_lev{lev}.weight"] = chunk
    # surface variables: old shape (D, frames, 1, 1) = new shape → direct copy
    return new_ckpt
```

If starting fresh (no prior checkpoint to migrate), train phase-1 from scratch with the
new embedder — the WoFS training set is large enough that the improved vertical structure
typically converges faster than the flat approach despite the parameter count increase.

---

## Summary of Recommended Changes (Priority Order)

| Priority | Change | Impact | Effort |
|---|---|---|---|
| **1 (High)** | Replace `VariableTokenEmbedder` with `VerticalLevelEmbedder` | Better vertical structure representation; per-level weight specialization; no backbone change | 1–2 days coding + testing |
| **2 (Medium)** | Add Fourier lat/lon positional encoding; remove position channels from `dyn_forcing` | Position not burned into input channels; additive encoding always active | 0.5 days |
| **3 (Medium)** | Additive vertical level encoding inside `VerticalLevelEmbedder` | Network distinguishes eta levels of same variable without index tricks | Already included in priority 1 |
| **4 (Low)** | Quadrant boundary pooling in FiLM | Coarse spatial boundary structure preserved | 0.5 days |
| **5 (Optional/Advanced)** | Per-frame Conv2d per level (instead of Conv2d(frames, D)) | Full Aurora-style temporal kernel | Requires T>1 training data; deferred |
| **Keep as is** | CrossFormer backbone, anti-checkerboard decoder, SDL noise injection, TensorPadding, VariableTokenDecoder, KCRPS training | These are already superior to Aurora for this task | — |
