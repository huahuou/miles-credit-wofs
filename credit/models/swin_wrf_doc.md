    
    
    
Yes, that is exactly correct. Here is the precise data flow that guarantees it for both resolutions:

---

## End-to-End Shape Trace

### 300×300 input

```
x input:          (B, Vars, T, 300, 300)
  ↓ TensorPadding (if enabled) — assume no boundary pad for clarity
  ↓ orig_H, orig_W = 300, 300  ← recorded here

CubeEmbedding (Conv3d, patch=4):
  → (B, dim, 1, 75, 75)        ← 300//4 = 75
  → .squeeze(2)
  → (B, dim, 75, 75)

UTransformer.DownBlock (stride=2):
  → (B, dim, 38, 38)           ← ceil(75/2) = 38

  dynamic pad to window_size=7:
    pad_b = (7 - 38%7)%7 = (7-3)%7 = 4
    pad_r = (7 - 38%7)%7 = 4
  → (B, dim, 42, 42)           ← window-divisible ✓

  SwinV2Stage (dynamic_mask=True, no hardcoded shape):
  → (B, 42, 42, dim)  →  (B, dim, 42, 42)

  crop back:  [:, :, :38, :38]
  → (B, dim, 38, 38)

  UpBlock (ConvTranspose2d stride=2):
  → (B, dim, 75, 75)           ← back to pre-DownBlock size ✓
  
  (actual output may be 76 due to ConvTranspose2d math;
   the crop in fix 3c handles this — see below)

fc + pixel-space reassembly (patch_lat=4, patch_lon=4):
  Lat_p=75, Lon_p=75 (read from live tensor)
  → (B, 75*4, 75*4) = (B, 300, 300)

FINAL CROP:  x[:, :, :300, :300]
output:       (B, out_chans, 300, 300)  ✓
```

---

### 900×900 input

```
x input:          (B, Vars, T, 900, 900)
  ↓ orig_H, orig_W = 900, 900  ← recorded here

CubeEmbedding (Conv3d, patch=4):
  → (B, dim, 1, 225, 225)      ← 900//4 = 225
  → .squeeze(2)
  → (B, dim, 225, 225)

UTransformer.DownBlock (stride=2):
  → (B, dim, 113, 113)         ← ceil(225/2) = 113

  dynamic pad to window_size=7:
    pad_b = (7 - 113%7)%7 = (7-1)%7 = 6
    pad_r = (7 - 113%7)%7 = 6
  → (B, dim, 119, 119)         ← window-divisible ✓

  SwinV2Stage (dynamic_mask=True, recomputes mask for 119×119):
  → (B, 119, 119, dim)  →  (B, dim, 119, 119)

  crop back:  [:, :, :113, :113]
  → (B, dim, 113, 113)

  UpBlock (ConvTranspose2d stride=2):
  → (B, dim, 226, 226)

fc + pixel-space reassembly (patch_lat=4, patch_lon=4):
  Lat_p=226, Lon_p=226 (read from live tensor)
  → (B, 226*4, 226*4) = (B, 904, 904)

FINAL CROP:  x[:, :, :900, :900]
output:       (B, out_chans, 900, 900)  ✓
```

---

## Why the Final Crop is the Safety Net

The `ConvTranspose2d(stride=2)` in `UpBlock` has the classic off-by-one ambiguity — an input of `H` can produce either `2H` or `2H-1` depending on `output_padding`. For odd input sizes like 75 or 113 the output is slightly larger than the original pre-embedding size. The final crop:

```python
x = x[:, :, :orig_H, :orig_W].contiguous()
```

handles this cleanly for **any** resolution without any special-casing. It is lossless because the extra pixels are always padding artefacts, never real data.

---

## Summary

| Resolution | Patch grid | After DownBlock | After UpBlock+reassembly | After crop |
|---|---|---|---|---|
| 300×300 | 75×75 | 38×38 | 300×300 (or 300) | **300×300** ✓ |
| 900×900 | 225×225 | 113×113 | 904×904 | **900×900** ✓ |

So yes — the model is a true input-output shape mirror for any resolution, and the same pretrained weights apply to both.



Great question — the answer depends on **how different 900×900 is from 300×300 in terms of what the model has already learned**. Let me break it down properly.

---

## What the model actually learned at 300×300

| Component | What it learned | Resolution-dependent? |
|---|---|---|
| `cube_embedding_inside.proj` (Conv3d) | Local patch statistics, variable correlations | ❌ No — Conv3d weights are spatial-local |
| `u_transformer.layer` (SwinV2 blocks) | Mesoscale patterns within each `7×7` window | ❌ No — CPB is window-local, not grid-global |
| `u_transformer.down/up` (Conv2d/ConvTranspose2d) | Multi-scale feature extraction | ❌ No — convolutional, fully transferable |
| `fc` (Linear) | How to map dim → pixel values | ❌ No — channel-only |
| `film` (Linear) | Time encoding → scale/shift | ❌ No — channel-only |
| `cpb_mlp` inside each attention block | Log-space relative positions within a window | ⚠️ Soft — position range is the same, but density of tokens per window changes |

**Nothing in the architecture is globally position-aware** (no absolute positional embeddings, no global attention). This is SwinV2's key advantage for your use case.

---

## The real concern: what changes at 900×900

At 300×300 with patch size 4, the model sees **75×75 patch tokens**. At 900×900 it sees **225×225 tokens** — 9× more. Each `7×7` window still covers the same **number of tokens**, but those tokens now represent a **physically smaller area** (1/3 the spatial extent per token).

This means:

- The **CPB** is fine — it only knows about relative positions *within* a window, and the window token count is identical
- The **DownBlock** now operates on a 9× denser feature map — convolutional weights transfer perfectly
- The **effective receptive field in physical space shrinks** — a 7-window at 300×300 covered ~112 km; at 900×900 it covers ~37 km

So the model isn't broken, but it will initially **underperform** at 900×900 because its learned mesoscale patterns encoded implicit assumptions about physical scale.

---

## Recommendation: Progressive Strategy

### Phase 1 — Warm up with frozen backbone (2–5% of total fine-tuning steps)

```python
# Freeze everything except the final projection head
for name, param in model.named_parameters():
    if "fc" not in name and "film" not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

**Why:** The 900×900 input produces a very different activation distribution. Letting the fc + film layers adapt first prevents the large gradient signal from a new resolution from immediately corrupting the SwinV2 attention weights.

---

### Phase 2 — Unfreeze with layer-wise learning rate decay (main fine-tuning)

```python
# Layer-wise LR decay: deeper (earlier) layers get smaller LR
# because they contain the most transferable general features

param_groups = [
    # Most transferable — smallest LR
    {
        "params": model.cube_embedding_inside.parameters(),
        "lr": 1e-5,
        "name": "cube_embedding",
    },
    # Core attention — small LR, but CPB needs to adapt to new token density
    {
        "params": model.u_transformer.layer.parameters(),
        "lr": 3e-5,
        "name": "swin_stage",
    },
    # Conv up/down — moderate LR
    {
        "params": [
            *model.u_transformer.down.parameters(),
            *model.u_transformer.up.parameters(),
        ],
        "lr": 5e-5,
        "name": "conv_blocks",
    },
    # Head layers — largest LR, need to adapt most
    {
        "params": [
            *model.fc.parameters(),
            *model.film.parameters(),
        ],
        "lr": 1e-4,
        "name": "head",
    },
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
```

---

### Phase 3 — Full fine-tuning at very low LR (optional, last 10–20% of steps)

```python
# Unfreeze everything, tiny LR across the board
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
```

---

## One thing you MUST do: `pretrained_window_size`

When you instantiate the fine-tuning model, pass the **original pretraining window size** into `SwinTransformerV2Stage` so the CPB MLP normalises coordinates correctly:

```python
# In UTransformer.__init__, change to:
self.layer = SwinTransformerV2Stage(
    ...
    pretrained_window_size=window_size[0],  # ← the 300x300 training window size
    dynamic_mask=True,
    always_partition=True,
)
```

This tells the CPB to normalise relative coordinates relative to the **pretrained** window size, exactly like the official `swinv2_base_window12to16` checkpoint does when going from 12→16 window size.

---

## Decision Summary

| Scenario | Strategy |
|---|---|
| Fast experiment / limited 900×900 data | Small global LR (`1e-5`) only, no freezing |
| Moderate 900×900 data | Phase 1 (freeze) + Phase 2 (layer-wise LR) |
| Large 900×900 dataset | All 3 phases, full fine-tuning |
| Very limited 900×900 data | Freeze everything except `fc` + `film` permanently |

The **single most impactful thing** you can do regardless of data size is **set `pretrained_window_size`** correctly — it costs nothing and immediately gives the CPB the right coordinate frame for transfer.