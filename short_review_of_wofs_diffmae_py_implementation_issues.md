# Copilot Chat Conversation Export: Review of wofs_diffmae.py implementation issues


## most faithful to DiffMAE implementation https://arxiv.org/pdf/2304.03283 
Strictly speaking, the most faithful version would do this:

- pass `precip_visible` through a **conditioning adapter**
- include it in `_condition_tokens(...)`
- only diffuse the masked target region
- avoid mixing visible precip into the sampled image state except as conditioning

That would require modifying more than just sampling functions.

So I’d describe the script above as:

- **much more faithful than current code**
- **clean visible region preserved**
- **sampling adapted to RePaint-DDIM**
- **still a sampler-level approximation of DiffMAE conditioning**, not a full architectural conversion

## My recommendation
Use the script above first. It fixes the memory bug and aligns sampling much better with your goals.

If you want, I can next provide a **second script** that also updates the model architecture so `precip_visible` becomes a true clean conditioning modality inside `_condition_tokens()` and `forward()`.

## @huahuou

provide a second script that also updates the model architecture so precip_visible becomes a true clean conditioning modality inside _condition_tokens() and forward()

## Copilot

Yes — below is a **second script** that updates the architecture so `precip_visible` becomes a **true clean conditioning modality** inside `_condition_tokens()` and `forward()`.

## What this script changes

It does four architectural things:

1. **Adds a visible-precip conditioning adapter**
   - `self.visible_precip_adapter`
   - same tokenization scale as target precip

2. **Adds a visible-precip modality embedding**
   - `self.visible_precip_modality_emb`

3. **Extends `_condition_tokens()`**
   - accepts `precip_visible`
   - tokenizes it as clean context
   - appends it to the condition token list

4. **Extends `forward()` path**
   - `_legacy_forward`, `_grouped_forward`, `_grouped_forward_joint`, and `forward`
   - all accept `precip_visible`
   - use it through `_condition_tokens(...)`

It also updates:
- `model_predictions`
- `p_mean_variance`
- `p_losses`
- `sample_precip`

so the new clean conditioning path is used consistently.

## Important design note
This is the **right architectural direction** if you want to be faithful to DiffMAE:

- `noisy_precip` remains the diffused target
- `precip_visible` is clean context
- visible precip is no longer merely “copied back” at sampling time; it is also provided as actual conditioning tokens

I still keep `_apply_visible_precip()` in the script, but in the clean-DiffMAE version it should act only as a **hard constraint on the known pixels during sampling**, not as the primary way the model sees visible precip.

---

# Script to replace/update functions in `wofs_diffmae.py`

```python name=wofs_diffmae_visible_precip_conditioning_patch.py
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from credit.models.wofs_mae_adapters import WoFSInputAdapter, trunc_normal_


def patched_init_visible_precip_conditioning(self):
    """
    Call this inside __init__ after condition_adapters is created and after
    target_pos_emb/target_modality_emb are initialized.
    """
    self.visible_precip_adapter = WoFSInputAdapter(
        num_channels=self.channels,
        patch_size=self.patch_size,
        embed_dim=self.embed_dim,
        image_size=self.image_size,
    )
    self.visible_precip_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    trunc_normal_(self.visible_precip_modality_emb, std=0.02)


def _condition_tokens(
    self,
    cond: Dict[str, torch.Tensor],
    precip_visible: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """
    Build clean conditioning tokens from:
      - existing condition modalities
      - optional clean visible precip tokens
    """
    cond_tokens = []

    for mod in self.conditioned_modalities:
        if mod in cond:
            cond_tokens.append(self.condition_adapters[mod](cond[mod]))

    if precip_visible is not None:
        visible_tokens = self.visible_precip_adapter(precip_visible)
        pos = rearrange(
            self.target_pos_emb.to(precip_visible.dtype),
            "b d h w -> b (h w) d",
        )
        visible_tokens = visible_tokens + pos + self.visible_precip_modality_emb
        cond_tokens.append(visible_tokens)

    return cond_tokens


def _legacy_forward(
    self,
    noisy_precip: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, _, orig_h, orig_w = noisy_precip.shape
    noisy_tokens = self.legacy_precip_adapter(noisy_precip)
    token_mask = self.token_mask_from_precip_mask(precip_mask).to(noisy_tokens.dtype).unsqueeze(-1)
    target_tokens = noisy_tokens + self.mask_token * token_mask
    pos = rearrange(self.target_pos_emb.to(noisy_precip.dtype), "b d h w -> b (h w) d")
    target_tokens = target_tokens + pos + self.target_modality_emb

    cond_tokens = self._condition_tokens(cond, precip_visible=precip_visible)
    tokens = self._decode_target_tokens(target_tokens, cond_tokens, t)
    patches = self.legacy_out_proj(tokens)

    n_h = self.image_size[0] // self.patch_size
    n_w = self.image_size[1] // self.patch_size
    x = patches.reshape(b, n_h, n_w, self.channels, self.patch_size, self.patch_size)
    x = torch.einsum("bhwcpq->bchpwq", x).reshape(
        b, self.channels, n_h * self.patch_size, n_w * self.patch_size
    )
    return self.anti_patch_refiner(x[:, :, :orig_h, :orig_w])


def _grouped_forward(
    self,
    noisy_precip: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, _, orig_h, orig_w = noisy_precip.shape
    cond_tokens = self._condition_tokens(cond, precip_visible=precip_visible)

    if self.grouped_decoder_scope == "joint":
        return self._grouped_forward_joint(
            noisy_precip, t, cond_tokens, precip_mask
        )

    outs = []
    for group_idx, (name, (start, stop)) in enumerate(
        zip(self.precip_group_names, self.precip_group_slices)
    ):
        group_tokens = self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
        tokens = self._decode_target_tokens(group_tokens, cond_tokens, t)
        patches = self.precip_group_out_proj[name](tokens)
        n_ch = stop - start
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
        group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(
            b, n_ch, n_h * self.patch_size, n_w * self.patch_size
        )
        outs.append(group_out[:, :, :orig_h, :orig_w])

    return self.anti_patch_refiner(torch.cat(outs, dim=1))


def _grouped_forward_joint(
    self,
    noisy_precip: torch.Tensor,
    t: torch.Tensor,
    cond_tokens: list[torch.Tensor],
    precip_mask: torch.Tensor,
) -> torch.Tensor:
    b, _, orig_h, orig_w = noisy_precip.shape
    group_tokens = [
        self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
        for group_idx, (name, (start, stop)) in enumerate(
            zip(self.precip_group_names, self.precip_group_slices)
        )
    ]
    n_tokens = group_tokens[0].shape[1]
    decoded = self._decode_target_tokens(torch.cat(group_tokens, dim=1), cond_tokens, t)
    decoded_groups = decoded.split(n_tokens, dim=1)

    outs = []
    for tokens, name, (start, stop) in zip(
        decoded_groups, self.precip_group_names, self.precip_group_slices
    ):
        patches = self.precip_group_out_proj[name](tokens)
        n_ch = stop - start
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
        group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(
            b, n_ch, n_h * self.patch_size, n_w * self.patch_size
        )
        outs.append(group_out[:, :, :orig_h, :orig_w])

    return self.anti_patch_refiner(torch.cat(outs, dim=1))


def forward(
    self,
    noisy_precip: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if self.grouped_precip:
        return self._grouped_forward(
            noisy_precip,
            t,
            cond,
            precip_mask,
            precip_visible=precip_visible,
        )
    return self._legacy_forward(
        noisy_precip,
        t,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )


def model_predictions(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
):
    model_out = self.forward(
        x_t,
        t,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )
    if self.objective == "pred_noise":
        pred_noise = model_out
        pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)
    elif self.objective == "pred_x0":
        pred_x_start = model_out
        pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
    else:
        pred_x_start = self.predict_start_from_v(x_t, t, model_out)
        pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
    return pred_noise, pred_x_start


def p_mean_variance(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
):
    _, x_start = self.model_predictions(
        x_t,
        t,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_start, x_t=x_t, t=t
    )
    return model_mean, posterior_variance, posterior_log_variance, x_start


def p_losses(
    self,
    x_start: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    b = x_start.shape[0]
    if t is None:
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = self.q_sample(x_start, t, noise)
    model_out = self.forward(
        x_t,
        t,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )

    if self.objective == "pred_noise":
        target = noise
    elif self.objective == "pred_x0":
        target = x_start
    else:
        target = self.predict_v(x_start, t, noise)

    pixel_mask = self.expand_patch_mask(
        precip_mask, x_start.shape[-2], x_start.shape[-1]
    ).to(x_start.dtype)
    loss_raw = (model_out - target) ** 2
    denom = pixel_mask.sum()
    if pixel_mask.shape[1] == 1:
        denom = denom * x_start.shape[1]
    loss = (loss_raw * pixel_mask).sum() / denom.clamp_min(1.0)

    return {
        "loss": loss,
        "model_out": model_out,
        "target": target,
        "x_t": x_t,
        "t": t,
    }


def _apply_visible_precip(
    self,
    img: torch.Tensor,
    precip_visible: Optional[torch.Tensor],
    pixel_mask: torch.Tensor,
    time: Optional[int],
) -> torch.Tensor:
    """
    Keep visible precip clean and fixed during sampling.
    With the architectural change, this is now a hard constraint,
    not the main conditioning mechanism.
    """
    del time
    if precip_visible is None:
        return img
    return img * pixel_mask + precip_visible * (1.0 - pixel_mask)


def _maybe_init_imgs(self, img: torch.Tensor, return_all_timesteps: bool):
    return [img] if return_all_timesteps else None


def _maybe_append_img(self, imgs, img: torch.Tensor) -> None:
    if imgs is not None:
        imgs.append(img)


def _finalize_imgs(self, imgs, img: torch.Tensor) -> torch.Tensor:
    if imgs is not None:
        return torch.stack(imgs, dim=1)
    return img


def _ddim_timesteps(self, total_steps: int) -> list[int]:
    total_steps = max(1, min(int(total_steps), self.num_timesteps))
    if total_steps == self.num_timesteps:
        return list(reversed(range(self.num_timesteps)))

    ts = torch.linspace(0, self.num_timesteps - 1, steps=total_steps, device=self.device)
    ts = torch.round(ts).long().tolist()

    deduped = []
    seen = set()
    for t in ts:
        t = int(t)
        if t not in seen:
            deduped.append(t)
            seen.add(t)

    return list(reversed(deduped))


def _ddim_predict_x0_eps(
    self,
    img: torch.Tensor,
    time: int,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
):
    b = img.shape[0]
    time_cond = torch.full((b,), time, device=img.device, dtype=torch.long)
    pred_noise, pred_x0 = self.model_predictions(
        img,
        time_cond,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )
    return pred_x0, pred_noise


def _ddim_step(
    self,
    img: torch.Tensor,
    time: int,
    time_next: int,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    eta: float,
    precip_visible: Optional[torch.Tensor] = None,
):
    pred_x0, pred_eps = self._ddim_predict_x0_eps(
        img,
        time,
        cond,
        precip_mask,
        precip_visible=precip_visible,
    )

    if time_next < 0:
        return pred_x0, pred_x0, pred_eps

    alpha_t = self.alphas_cumprod[time]
    alpha_next = self.alphas_cumprod[time_next]

    sigma = eta * torch.sqrt(
        torch.clamp(
            (1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next),
            min=0.0,
        )
    )
    c = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma**2, min=0.0))

    noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
    img_next = torch.sqrt(alpha_next) * pred_x0 + c * pred_eps + sigma * noise
    return img_next, pred_x0, pred_eps


@torch.no_grad()
def sample_precip(
    self,
    cond: Dict[str, torch.Tensor],
    precip_mask: torch.Tensor,
    precip_visible: Optional[torch.Tensor] = None,
    sampling_timesteps: Optional[int] = None,
    eta: Optional[float] = None,
    sampler: str = "ddim",
    repaint_jump_length: int = 10,
    repaint_jump_n_sample: int = 10,
    return_all_timesteps: bool = False,
) -> torch.Tensor:
    first = next(iter(cond.values()))
    b, _, h, w = first.shape
    shape = (b, self.channels, h, w)

    sampler = str(sampler).strip().lower()
    if sampler not in {"ddim", "ddpm", "repaint", "repaint_ddim"}:
        raise ValueError(f"Unsupported DiffMAE sampler: {sampler!r}")

    total_steps = (
        self.num_timesteps
        if sampler in {"ddpm", "repaint"}
        else (sampling_timesteps or self.sampling_timesteps)
    )
    eta = self.ddim_sampling_eta if eta is None else float(eta)

    img = torch.randn(shape, device=first.device, dtype=first.dtype)
    pixel_mask = self.expand_patch_mask(precip_mask, h, w).to(first.dtype)

    start_time = self.num_timesteps - 1
    img = self._apply_visible_precip(img, precip_visible, pixel_mask, start_time)
    imgs = self._maybe_init_imgs(img, return_all_timesteps)

    if sampler == "ddpm":
        for time in reversed(range(self.num_timesteps)):
            time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
            model_mean, _, model_log_variance, _ = self.p_mean_variance(
                img,
                time_cond,
                cond,
                precip_mask,
                precip_visible=precip_visible,
            )
            noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
            img = model_mean + (0.5 * model_log_variance).exp() * noise
            img = self._apply_visible_precip(img, precip_visible, pixel_mask, time - 1)
            self._maybe_append_img(imgs, img)
        return self._finalize_imgs(imgs, img)

    if sampler == "repaint":
        times = self._repaint_times(start_time, repaint_jump_length, repaint_jump_n_sample)
        if not times or times[0] != start_time:
            raise RuntimeError(
                f"Invalid RePaint schedule starts with {times[0] if times else None}, expected {start_time}"
            )

        for time, time_next in zip(times[:-1], times[1:]):
            if time_next > time:
                img = self._repaint_forward_step(img, time, time_next)
                img = self._apply_visible_precip(img, precip_visible, pixel_mask, time_next)
                self._maybe_append_img(imgs, img)
                continue

            time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
            model_mean, _, model_log_variance, _ = self.p_mean_variance(
                img,
                time_cond,
                cond,
                precip_mask,
                precip_visible=precip_visible,
            )
            noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
            img = model_mean + (0.5 * model_log_variance).exp() * noise
            img = self._apply_visible_precip(img, precip_visible, pixel_mask, time_next)
            self._maybe_append_img(imgs, img)

        return self._finalize_imgs(imgs, img)

    if sampler == "repaint_ddim":
        times = self._ddim_timesteps(total_steps)
        resample_u = max(1, int(repaint_jump_n_sample))

        for idx, time in enumerate(times):
            time_next = times[idx + 1] if idx + 1 < len(times) else -1

            pred_x0 = None
            pred_eps = None

            for u in range(resample_u):
                pred_x0, pred_eps = self._ddim_predict_x0_eps(
                    img,
                    time,
                    cond,
                    precip_mask,
                    precip_visible=precip_visible,
                )

                if precip_visible is not None:
                    pred_x0 = pred_x0 * pixel_mask + precip_visible * (1.0 - pixel_mask)

                if u < resample_u - 1:
                    alpha_t = self.alphas_cumprod[time]
                    eps = torch.randn_like(img)
                    img = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(1.0 - alpha_t) * eps
                    img = self._apply_visible_precip(img, precip_visible, pixel_mask, time)

            if time_next < 0:
                img = pred_x0
            else:
                alpha_next = self.alphas_cumprod[time_next]
                img = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * pred_eps
                img = self._apply_visible_precip(img, precip_visible, pixel_mask, time_next)

            self._maybe_append_img(imgs, img)

        return self._finalize_imgs(imgs, img)

    times = self._ddim_timesteps(total_steps)
    times_next = times[1:] + [-1]

    for time, time_next in zip(times, times_next):
        img, pred_x0, _ = self._ddim_step(
            img,
            time,
            time_next,
            cond,
            precip_mask,
            eta,
            precip_visible=precip_visible,
        )

        if precip_visible is not None:
            if time_next < 0:
                img = pred_x0 * pixel_mask + precip_visible * (1.0 - pixel_mask)
            else:
                img = self._apply_visible_precip(img, precip_visible, pixel_mask, time_next)

        self._maybe_append_img(imgs, img)

    return self._finalize_imgs(imgs, img)
```

---

## How to apply this script

You should make these changes inside `WoFSDiffMAE.__init__` too.

### 1. Add this in `__init__` after `self.condition_adapters` are built
Use this near where the condition adapters are created in `wofs_diffmae.py`. The relevant current block is around here: `credit/models/wofs_diffmae.py#L292-L301`.

Add:

```python name=__init__ addition snippet
self.visible_precip_adapter = WoFSInputAdapter(
    num_channels=self.channels,
    patch_size=self.patch_size,
    embed_dim=self.embed_dim,
    image_size=self.image_size,
)
```

### 2. Add this in `__init__` after `self.target_modality_emb`
The relevant current block is around `credit/models/wofs_diffmae.py#L352-L356`.

Add:

```python name=__init__ addition snippet
self.visible_precip_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
trunc_normal_(self.visible_precip_modality_emb, std=0.02)
```

---

## What this gives you conceptually

After this patch, `precip_visible` is used in **two ways**:

1. **As clean context tokens** via `_condition_tokens(...)`
2. **As a hard sampling constraint** via `_apply_visible_precip(...)`

That is much better than the previous setup, where visible precip was only pasted into the sample and even re-noised.

## If you want to be even more strict about DiffMAE
The next stricter step would be:

- remove `_apply_visible_precip()` entirely from DDIM/DDPM sampling
- rely only on visible precip as conditioning tokens
- only generate masked target region implicitly through the mask-aware loss/training

But for inpainting tasks, I think keeping the hard visible-region constraint is practical and usually beneficial.

## One caution
Because I could not inspect `credit/models/wofs_mae_adapters.py` directly from the tool, I am assuming `WoFSInputAdapter` can be instantiated exactly the same way as elsewhere in `wofs_diffmae.py`, which is very likely true because that file already does so repeatedly.

If you want, I can next provide a **single consolidated full replacement of the entire `WoFSDiffMAE` class** so you don’t have to manually splice together the two scripts.

## @huahuou

I have upload the wofs_mae_adapters.py for you. Next can you provide the entire full wofs_diffmae.py script with all the modification inplace and go with the strict option use in the DiffMAE. 
remove _apply_visible_precip() entirely from DDIM/DDPM sampling
rely only on visible precip as conditioning tokens
only generate masked target region implicitly through the mask-aware loss/training

## Copilot

Yes — below is a **full replacement `wofs_diffmae.py`** that applies the strict DiffMAE-style choice you asked for:

- **`precip_visible` is a true clean conditioning modality**
- **no `_apply_visible_precip()` in DDIM/DDPM/RePaint/RePaint-DDIM sampling**
- **visible precip is only used through conditioning tokens**
- **generation is still targeted to masked regions through the mask-aware training loss**
- **the VRAM bug is fixed** by storing intermediate steps only when `return_all_timesteps=True`

One important note up front:

## Important caveat
With the strict option, the sampler no longer hard-clamps known visible pixels during inference. So:

- this is **more faithful to DiffMAE**
- but it is **less like hard inpainting**
- the model must learn to preserve/use visible information through the conditioning path alone

That is exactly what you asked for.

---

````python name=wofs_diffmae.py
"""Conditional DiffMAE for WoFS precip random-mask inpainting."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from credit.models.base_model import BaseModel
from credit.models.wofs_mae_adapters import (
    Attention,
    Block,
    CrossAttention,
    Mlp,
    WoFSInputAdapter,
    build_2d_sincos_posemb,
    trunc_normal_,
)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Iterable[int]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(tuple(x_shape)) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((x * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


class CrossSelfDecoderBlock(nn.Module):
    """DiffMAE decoder block: target tokens cross-attend context, then self-attend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        target_grid_size: Optional[Tuple[int, int]] = None,
        target_window_size: int = 0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_cross_q = norm_layer(dim)
        self.norm_cross_ctx = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm_self = norm_layer(dim)
        if target_window_size and target_window_size > 0:
            if target_grid_size is None:
                raise ValueError("target_grid_size is required when target_window_size > 0")
            self.self_attn = WindowAttention2D(
                dim,
                num_heads=num_heads,
                grid_size=target_grid_size,
                window_size=int(target_window_size),
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.self_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path_prob = float(drop_path)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob == 0.0:
            return x
        keep = 1.0 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[1] > 0:
            x = x + self._drop_path(self.cross_attn(self.norm_cross_q(x), self.norm_cross_ctx(context)))
        x = x + self._drop_path(self.self_attn(self.norm_self(x)))
        x = x + self._drop_path(self.mlp(self.norm_mlp(x)))
        return x


class WindowAttention2D(nn.Module):
    """Local self-attention over a 2-D target-token grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Tuple[int, int],
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        grid_h, grid_w = self.grid_size
        if n != grid_h * grid_w:
            raise ValueError(f"WindowAttention2D expected {grid_h * grid_w} tokens, got {n}")
        w = self.window_size
        x_grid = x.reshape(b, grid_h, grid_w, d)
        pad_h = (w - grid_h % w) % w
        pad_w = (w - grid_w % w) % w
        if pad_h or pad_w:
            x_grid = F.pad(x_grid, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x_grid.shape[1], x_grid.shape[2]
        x_windows = x_grid.reshape(b, padded_h // w, w, padded_w // w, w, d)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, d)
        x_windows = self.attn(x_windows)
        x_grid = x_windows.reshape(b, padded_h // w, padded_w // w, w, w, d)
        x_grid = x_grid.permute(0, 1, 3, 2, 4, 5).reshape(b, padded_h, padded_w, d)
        return x_grid[:, :grid_h, :grid_w].reshape(b, n, d)


class ResidualConvRefiner(nn.Module):
    """Small zero-initialized residual CNN for reducing patch-edge artifacts."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        depth: int = 2,
        kernel_size: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        depth = max(1, int(depth))
        hidden_channels = int(hidden_channels)
        kernel_size = int(kernel_size)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = int(channels)
        for _ in range(depth - 1):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=int(groups) if in_channels == hidden_channels else 1,
                )
            )
            layers.append(nn.GELU())
            in_channels = hidden_channels
        final = nn.Conv2d(in_channels, int(channels), kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class WoFSDiffMAE(BaseModel):
    """Diffusion model that inpaints normalized precip conditioned on unmasked WoFS context."""

    def __init__(
        self,
        modality_channels: Optional[Dict[str, int]] = None,
        conditioned_modalities: Optional[list[str]] = None,
        target_modality: str = "precip",
        precip_grouping: str = "legacy",
        decoder_type: str = "concat_self",
        grouped_decoder_scope: str = "per_group",
        precip_group_names: Optional[list[str]] = None,
        precip_group_channels: Optional[list[int]] = None,
        patch_size: int = 8,
        surface_forcing_stride: int = 16,
        image_size: Tuple[int, int] = (304, 304),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        target_attention_window_size: int = 0,
        anti_patch_refiner: Optional[dict] = None,
        diffusion: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.modality_channels = modality_channels or {
            "background": 102,
            "precip": 136,
            "reflectivity": 17,
            "surface": 2,
            "forcing": 10,
        }
        self.conditioned_modalities = conditioned_modalities or ["background", "surface", "forcing", "reflectivity"]
        self.target_modality = target_modality
        self.precip_grouping = str(precip_grouping).strip().lower()
        self.grouped_precip = self.precip_grouping in {"grouped", "by_variable", "factorized"}
        self.decoder_type = str(decoder_type).strip().lower()
        if self.decoder_type not in {"concat_self", "cross_self"}:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.grouped_decoder_scope = str(grouped_decoder_scope).strip().lower()
        if self.grouped_decoder_scope not in {"per_group", "joint"}:
            raise ValueError(f"Unsupported grouped_decoder_scope: {grouped_decoder_scope}")
        self.precip_group_names = list(precip_group_names or [])
        self.precip_group_channels = list(precip_group_channels or [])
        self.patch_size = int(patch_size)
        self.surface_forcing_stride = int(surface_forcing_stride)
        self.image_size = tuple(image_size)
        self.embed_dim = int(embed_dim)
        self.channels = int(self.modality_channels[target_modality])
        self.output_channels = self.channels
        self.input_channels = self.channels
        self.frames = 1
        self.self_condition = False
        self.condition = True
        self.target_grid_size = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        self.target_attention_window_size = int(target_attention_window_size)

        refiner_conf = anti_patch_refiner or {}
        self.anti_patch_refiner_enabled = bool(refiner_conf.get("enabled", False))
        self.anti_patch_refiner = nn.Identity()
        if self.anti_patch_refiner_enabled:
            self.anti_patch_refiner = ResidualConvRefiner(
                channels=self.channels,
                hidden_channels=int(refiner_conf.get("hidden_channels", 64)),
                depth=int(refiner_conf.get("depth", 2)),
                kernel_size=int(refiner_conf.get("kernel_size", 3)),
                groups=int(refiner_conf.get("groups", 1)),
            )

        sf_h = math.ceil(self.image_size[0] / self.surface_forcing_stride) * self.surface_forcing_stride
        sf_w = math.ceil(self.image_size[1] / self.surface_forcing_stride) * self.surface_forcing_stride
        sf_image_size = (sf_h, sf_w)

        self.condition_adapters = nn.ModuleDict()
        for mod in self.conditioned_modalities:
            stride = self.surface_forcing_stride if mod in {"surface", "forcing"} else self.patch_size
            adapter_image_size = sf_image_size if mod in {"surface", "forcing"} else self.image_size
            self.condition_adapters[mod] = WoFSInputAdapter(
                num_channels=self.modality_channels[mod],
                patch_size=stride,
                embed_dim=self.embed_dim,
                image_size=adapter_image_size,
            )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        time_dim = self.embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.embed_dim),
            nn.Linear(self.embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, self.embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if self.decoder_type == "cross_self":
            self.blocks = nn.ModuleList(
                [
                    CrossSelfDecoderBlock(
                        self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[i],
                        target_grid_size=self.target_grid_size,
                        target_window_size=self.target_attention_window_size,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [Block(self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(depth)]
            )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.legacy_precip_adapter = None
        self.legacy_out_proj = None
        self.precip_group_slices: list[Tuple[int, int]] = []
        self.precip_group_adapters = nn.ModuleDict()
        self.precip_group_out_proj = nn.ModuleDict()
        self.precip_group_embeds = nn.ParameterDict()
        if self.grouped_precip:
            self._init_grouped_precip_tokenization()
        else:
            self.legacy_precip_adapter = WoFSInputAdapter(
                num_channels=self.channels,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
            )
            self.legacy_out_proj = nn.Linear(self.embed_dim, self.channels * self.patch_size * self.patch_size)

        n_h, n_w = self.target_grid_size
        pos_emb = build_2d_sincos_posemb(n_h, n_w, self.embed_dim)
        self.register_buffer("target_pos_emb", pos_emb)
        self.target_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.target_modality_emb, std=0.02)

        # strict DiffMAE-style clean visible precip conditioning
        self.visible_precip_adapter = WoFSInputAdapter(
            num_channels=self.channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            image_size=self.image_size,
        )
        self.visible_precip_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.visible_precip_modality_emb, std=0.02)

        diffusion_conf = diffusion or {}
        self.objective = diffusion_conf.get("objective", "pred_v")
        if self.objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"Unsupported diffusion objective: {self.objective}")
        self.num_timesteps = int(diffusion_conf.get("timesteps", 1000))
        self.sampling_timesteps = int(diffusion_conf.get("sampling_timesteps", self.num_timesteps))
        self.ddim_sampling_eta = float(diffusion_conf.get("ddim_sampling_eta", 0.0))
        beta_schedule = diffusion_conf.get("beta_schedule", "sigmoid")
        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        self._register_diffusion_buffers(betas)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _register_diffusion_buffers(self, betas: torch.Tensor) -> None:
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float())
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)).float())
        self.register_buffer("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())
        self.register_buffer("posterior_mean_coef2", ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float())

    def _init_grouped_precip_tokenization(self) -> None:
        if not self.precip_group_names:
            raise ValueError("precip_group_names must be provided when precip_grouping is grouped")
        if not self.precip_group_channels:
            if self.channels % len(self.precip_group_names) != 0:
                raise ValueError(
                    "precip_group_channels must be provided unless precip channels divide evenly "
                    "across precip_group_names"
                )
            group_size = self.channels // len(self.precip_group_names)
            self.precip_group_channels = [group_size] * len(self.precip_group_names)
        if len(self.precip_group_names) != len(self.precip_group_channels):
            raise ValueError("precip_group_names and precip_group_channels must have the same length")
        if sum(self.precip_group_channels) != self.channels:
            raise ValueError(
                f"Grouped precip channels sum to {sum(self.precip_group_channels)} but target has {self.channels}"
            )

        start = 0
        for name, n_ch in zip(self.precip_group_names, self.precip_group_channels):
            stop = start + int(n_ch)
            self.precip_group_slices.append((start, stop))
            self.precip_group_adapters[name] = WoFSInputAdapter(
                num_channels=int(n_ch),
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
            )
            self.precip_group_out_proj[name] = nn.Linear(self.embed_dim, int(n_ch) * self.patch_size * self.patch_size)
            emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(emb, std=0.02)
            self.precip_group_embeds[name] = emb
            start = stop

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask[:, None]
        pooled = F.max_pool2d(mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        return rearrange(pooled, "b 1 h w -> b (h w)")

    def expand_patch_mask(self, patch_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        if patch_mask.dim() == 4:
            if not self.grouped_precip:
                raise ValueError("4-D height masks require grouped precip tokenization")
            b, n_groups, _, _ = patch_mask.shape
            if n_groups != len(self.precip_group_slices):
                raise ValueError(
                    f"Height mask has {n_groups} groups but model has {len(self.precip_group_slices)} precip groups"
                )
            expanded = []
            for group_idx, (start, stop) in enumerate(self.precip_group_slices):
                n_ch = stop - start
                group_mask = patch_mask[:, group_idx, :n_ch, :]
                if group_mask.shape[1] != n_ch:
                    raise ValueError(
                        f"Height mask group {group_idx} has {group_mask.shape[1]} levels, expected {n_ch}"
                    )
                expanded.append(group_mask.reshape(b, n_ch, n_h, n_w))
            mask = torch.cat(expanded, dim=1)
        elif patch_mask.dim() == 3:
            mask = patch_mask.reshape(patch_mask.shape[0], patch_mask.shape[1], n_h, n_w)
            if self.grouped_precip and patch_mask.shape[1] == len(self.precip_group_slices):
                mask = torch.cat(
                    [
                        mask[:, group_idx: group_idx + 1].expand(-1, stop - start, -1, -1)
                        for group_idx, (start, stop) in enumerate(self.precip_group_slices)
                    ],
                    dim=1,
                )
        else:
            mask = patch_mask.reshape(patch_mask.shape[0], 1, n_h, n_w)
        mask = mask.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
        return mask[:, :, :height, :width]

    def precip_group_index_for_channel(self, channel_idx: int) -> int:
        if not self.grouped_precip:
            return 0
        for group_idx, (start, stop) in enumerate(self.precip_group_slices):
            if start <= channel_idx < stop:
                return group_idx
        raise IndexError(f"channel index {channel_idx} outside grouped precip slices")

    def random_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_tokens).long().clamp(0, n_tokens)
        return (ranks < n_mask[:, None]).float()

    def random_channel_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        if self.grouped_precip:
            return self.random_group_precip_mask(batch_size, mask_ratio, device)
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        n_elements = self.channels * n_tokens
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_elements, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_elements).long().clamp(0, n_elements)
        return (ranks < n_mask[:, None]).float().reshape(batch_size, self.channels, n_tokens)

    def random_group_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        n_groups = len(self.precip_group_slices) if self.grouped_precip else self.channels
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_groups * n_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * (n_groups * n_tokens)).long().clamp(0, n_groups * n_tokens)
        return (ranks < n_mask[:, None]).float().reshape(batch_size, n_groups, n_tokens)

    def random_height_precip_mask(
        self,
        batch_size: int,
        mask_ratio: float | tuple[float, float],
        device: torch.device,
        masked_levels: Optional[list[int]] = None,
        visible_levels: Optional[list[int]] = None,
    ) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        if not self.grouped_precip:
            return self.random_channel_precip_mask(batch_size, mask_ratio, device)
        group_channels = [stop - start for start, stop in self.precip_group_slices]
        if len(set(group_channels)) != 1:
            raise ValueError("Height masks require all precip groups to have the same number of levels")
        n_groups = len(group_channels)
        n_levels = group_channels[0]
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)

        eligible = torch.ones(n_levels, device=device, dtype=torch.bool)
        if masked_levels is not None:
            eligible = torch.zeros(n_levels, device=device, dtype=torch.bool)
            for level in masked_levels:
                if 0 <= int(level) < n_levels:
                    eligible[int(level)] = True
        if visible_levels is not None:
            for level in visible_levels:
                if 0 <= int(level) < n_levels:
                    eligible[int(level)] = False
        n_eligible = int(eligible.sum().item())
        mask = torch.zeros(batch_size, n_groups, n_levels, n_tokens, device=device)
        if n_eligible == 0:
            return mask

        n_elements = n_groups * n_eligible * n_tokens
        noise = torch.rand(batch_size, n_elements, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_elements).long().clamp(0, n_elements)
        sampled = (ranks < n_mask[:, None]).float().reshape(batch_size, n_groups, n_eligible, n_tokens)
        mask[:, :, eligible, :] = sampled
        return mask

    def token_mask_from_precip_mask(self, precip_mask: torch.Tensor) -> torch.Tensor:
        if precip_mask.dim() == 4:
            return precip_mask.amax(dim=(1, 2))
        if precip_mask.dim() == 3:
            return precip_mask.amax(dim=1)
        return precip_mask

    def _condition_tokens(
        self,
        cond: Dict[str, torch.Tensor],
        precip_visible: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        cond_tokens = []
        for mod in self.conditioned_modalities:
            if mod in cond:
                cond_tokens.append(self.condition_adapters[mod](cond[mod]))

        if precip_visible is not None:
            visible_tokens = self.visible_precip_adapter(precip_visible)
            pos = rearrange(self.target_pos_emb.to(precip_visible.dtype), "b d h w -> b (h w) d")
            visible_tokens = visible_tokens + pos + self.visible_precip_modality_emb
            cond_tokens.append(visible_tokens)

        return cond_tokens

    def _legacy_forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        noisy_tokens = self.legacy_precip_adapter(noisy_precip)
        token_mask = self.token_mask_from_precip_mask(precip_mask).to(noisy_tokens.dtype).unsqueeze(-1)
        target_tokens = noisy_tokens + self.mask_token * token_mask
        pos = rearrange(self.target_pos_emb.to(noisy_precip.dtype), "b d h w -> b (h w) d")
        target_tokens = target_tokens + pos + self.target_modality_emb

        cond_tokens = self._condition_tokens(cond, precip_visible=precip_visible)
        tokens = self._decode_target_tokens(target_tokens, cond_tokens, t)
        patches = self.legacy_out_proj(tokens)
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        x = patches.reshape(b, n_h, n_w, self.channels, self.patch_size, self.patch_size)
        x = torch.einsum("bhwcpq->bchpwq", x).reshape(b, self.channels, n_h * self.patch_size, n_w * self.patch_size)
        return self.anti_patch_refiner(x[:, :, :orig_h, :orig_w])

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        time_tokens = self.time_mlp(t).unsqueeze(1)
        if self.decoder_type == "cross_self":
            target_tokens = target_tokens + time_tokens
            context = target_tokens.new_zeros(target_tokens.shape[0], 0, target_tokens.shape[-1])
            if cond_tokens:
                context = torch.cat(cond_tokens, dim=1)
            for block in self.blocks:
                target_tokens = block(target_tokens, context)
            return self.norm(target_tokens)

        tokens = torch.cat([target_tokens] + cond_tokens, dim=1)
        tokens = tokens + time_tokens
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, : target_tokens.shape[1]])

    def _grouped_forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        cond_tokens = self._condition_tokens(cond, precip_visible=precip_visible)
        if self.grouped_decoder_scope == "joint":
            return self._grouped_forward_joint(noisy_precip, t, cond_tokens, precip_mask)

        outs = []
        for group_idx, (name, (start, stop)) in enumerate(zip(self.precip_group_names, self.precip_group_slices)):
            group_tokens = self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
            tokens = self._decode_target_tokens(group_tokens, cond_tokens, t)
            patches = self.precip_group_out_proj[name](tokens)
            n_ch = stop - start
            n_h = self.image_size[0] // self.patch_size
            n_w = self.image_size[1] // self.patch_size
            group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
            group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(b, n_ch, n_h * self.patch_size, n_w * self.patch_size)
            outs.append(group_out[:, :, :orig_h, :orig_w])
        return self.anti_patch_refiner(torch.cat(outs, dim=1))

    def _group_tokens(
        self,
        noisy_precip: torch.Tensor,
        precip_mask: torch.Tensor,
        group_idx: int,
        name: str,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        group_x = noisy_precip[:, start:stop]
        group_tokens = self.precip_group_adapters[name](group_x)
        if precip_mask.dim() == 2:
            group_mask = precip_mask
        elif precip_mask.dim() == 3:
            group_mask = precip_mask[:, group_idx, :]
        elif precip_mask.dim() == 4:
            group_mask = precip_mask[:, group_idx].amax(dim=1)
        else:
            raise ValueError(f"Unsupported precip_mask shape {tuple(precip_mask.shape)}")
        token_mask = group_mask.to(group_tokens.dtype).unsqueeze(-1)
        group_tokens = group_tokens + self.mask_token * token_mask
        pos = rearrange(self.target_pos_emb.to(noisy_precip.dtype), "b d h w -> b (h w) d")
        return group_tokens + pos + self.target_modality_emb + self.precip_group_embeds[name]

    def _grouped_forward_joint(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        group_tokens = [
            self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
            for group_idx, (name, (start, stop)) in enumerate(zip(self.precip_group_names, self.precip_group_slices))
        ]
        n_tokens = group_tokens[0].shape[1]
        decoded = self._decode_target_tokens(torch.cat(group_tokens, dim=1), cond_tokens, t)
        decoded_groups = decoded.split(n_tokens, dim=1)

        outs = []
        for tokens, name, (start, stop) in zip(decoded_groups, self.precip_group_names, self.precip_group_slices):
            patches = self.precip_group_out_proj[name](tokens)
            n_ch = stop - start
            n_h = self.image_size[0] // self.patch_size
            n_w = self.image_size[1] // self.patch_size
            group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
            group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(b, n_ch, n_h * self.patch_size, n_w * self.patch_size)
            outs.append(group_out[:, :, :orig_h, :orig_w])
        return self.anti_patch_refiner(torch.cat(outs, dim=1))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ):
        model_out = self.forward(x_t, t, cond, precip_mask, precip_visible=precip_visible)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == "pred_x0":
            pred_x_start = model_out
            pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
        else:
            pred_x_start = self.predict_start_from_v(x_t, t, model_out)
            pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
        return pred_noise, pred_x_start

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ):
        _, x_start = self.model_predictions(
            x_t,
            t,
            cond,
            precip_mask,
            precip_visible=precip_visible,
        )
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_out = self.forward(
            x_t,
            t,
            cond,
            precip_mask,
            precip_visible=precip_visible,
        )
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)
        pixel_mask = self.expand_patch_mask(precip_mask, x_start.shape[-2], x_start.shape[-1]).to(x_start.dtype)
        loss_raw = (model_out - target) ** 2
        denom = pixel_mask.sum()
        if pixel_mask.shape[1] == 1:
            denom = denom * x_start.shape[1]
        loss = (loss_raw * pixel_mask).sum() / denom.clamp_min(1.0)
        return {"loss": loss, "model_out": model_out, "target": target, "x_t": x_t, "t": t}

    def forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.grouped_precip:
            return self._grouped_forward(noisy_precip, t, cond, precip_mask, precip_visible=precip_visible)
        return self._legacy_forward(noisy_precip, t, cond, precip_mask, precip_visible=precip_visible)

    def _maybe_init_imgs(self, img: torch.Tensor, return_all_timesteps: bool):
        return [img] if return_all_timesteps else None

    def _maybe_append_img(self, imgs, img: torch.Tensor) -> None:
        if imgs is not None:
            imgs.append(img)

    def _finalize_imgs(self, imgs, img: torch.Tensor) -> torch.Tensor:
        if imgs is not None:
            return torch.stack(imgs, dim=1)
        return img

    def _ddim_timesteps(self, total_steps: int) -> list[int]:
        total_steps = max(1, min(int(total_steps), self.num_timesteps))
        if total_steps == self.num_timesteps:
            return list(reversed(range(self.num_timesteps)))

        ts = torch.linspace(0, self.num_timesteps - 1, steps=total_steps, device=self.device)
        ts = torch.round(ts).long().tolist()

        deduped = []
        seen = set()
        for t in ts:
            t = int(t)
            if t not in seen:
                deduped.append(t)
                seen.add(t)

        return list(reversed(deduped))

    def _ddim_predict_x0_eps(
        self,
        img: torch.Tensor,
        time: int,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ):
        b = img.shape[0]
        time_cond = torch.full((b,), time, device=img.device, dtype=torch.long)
        pred_noise, pred_x0 = self.model_predictions(
            img,
            time_cond,
            cond,
            precip_mask,
            precip_visible=precip_visible,
        )
        return pred_x0, pred_noise

    def _ddim_step(
        self,
        img: torch.Tensor,
        time: int,
        time_next: int,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        eta: float,
        precip_visible: Optional[torch.Tensor] = None,
    ):
        pred_x0, pred_eps = self._ddim_predict_x0_eps(
            img,
            time,
            cond,
            precip_mask,
            precip_visible=precip_visible,
        )

        if time_next < 0:
            return pred_x0, pred_x0, pred_eps

        alpha_t = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next),
                min=0.0,
            )
        )
        c = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma**2, min=0.0))

        noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
        img_next = torch.sqrt(alpha_next) * pred_x0 + c * pred_eps + sigma * noise
        return img_next, pred_x0, pred_eps

    def _repaint_times(self, start_time: int, jump_length: int, jump_n_sample: int) -> list[int]:
        jump_length = max(1, int(jump_length))
        jump_n_sample = max(1, int(jump_n_sample))
        t_t = int(start_time) + 1
        jumps = {
            time: jump_n_sample - 1
            for time in range(0, t_t - jump_length, jump_length)
        }
        times = []
        time = t_t
        while time >= 1:
            time -= 1
            times.append(time)
            if jumps.get(time, 0) > 0:
                jumps[time] -= 1
                for _ in range(jump_length):
                    time += 1
                    times.append(time)
        times.append(-1)
        return times

    def _repaint_forward_step(self, img: torch.Tensor, from_time: int, to_time: int) -> torch.Tensor:
        if to_time <= from_time:
            return img
        for time in range(from_time + 1, to_time + 1):
            beta = self.betas[time]
            img = (1.0 - beta).sqrt() * img + beta.sqrt() * torch.randn_like(img)
        return img

    @torch.no_grad()
    def sample_precip(
        self,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        sampling_timesteps: Optional[int] = None,
        eta: Optional[float] = None,
        sampler: str = "ddim",
        repaint_jump_length: int = 10,
        repaint_jump_n_sample: int = 10,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        first = next(iter(cond.values()))
        b, _, h, w = first.shape
        shape = (b, self.channels, h, w)

        sampler = str(sampler).strip().lower()
        if sampler not in {"ddim", "ddpm", "repaint", "repaint_ddim"}:
            raise ValueError(f"Unsupported DiffMAE sampler: {sampler!r}")

        total_steps = self.num_timesteps if sampler in {"ddpm", "repaint"} else (sampling_timesteps or self.sampling_timesteps)
        eta = self.ddim_sampling_eta if eta is None else float(eta)

        img = torch.randn(shape, device=first.device, dtype=first.dtype)
        imgs = self._maybe_init_imgs(img, return_all_timesteps)

        if sampler == "ddpm":
            for time in reversed(range(self.num_timesteps)):
                time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
                model_mean, _, model_log_variance, _ = self.p_mean_variance(
                    img,
                    time_cond,
                    cond,
                    precip_mask,
                    precip_visible=precip_visible,
                )
                noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
                img = model_mean + (0.5 * model_log_variance).exp() * noise
                self._maybe_append_img(imgs, img)
            return self._finalize_imgs(imgs, img)

        if sampler == "repaint":
            start_time = self.num_timesteps - 1
            if sampling_timesteps is not None:
                start_time = min(start_time, max(0, int(sampling_timesteps) - 1))

            times = self._repaint_times(start_time, repaint_jump_length, repaint_jump_n_sample)
            if not times or times[0] != start_time:
                raise RuntimeError(
                    f"Invalid RePaint schedule starts with {times[0] if times else None}, expected {start_time}"
                )

            for time, time_next in zip(times[:-1], times[1:]):
                if time_next > time:
                    img = self._repaint_forward_step(img, time, time_next)
                    self._maybe_append_img(imgs, img)
                    continue

                time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
                model_mean, _, model_log_variance, _ = self.p_mean_variance(
                    img,
                    time_cond,
                    cond,
                    precip_mask,
                    precip_visible=precip_visible,
                )
                noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
                img = model_mean + (0.5 * model_log_variance).exp() * noise
                self._maybe_append_img(imgs, img)

            return self._finalize_imgs(imgs, img)

        if sampler == "repaint_ddim":
            times = self._ddim_timesteps(total_steps)
            resample_u = max(1, int(repaint_jump_n_sample))

            for idx, time in enumerate(times):
                time_next = times[idx + 1] if idx + 1 < len(times) else -1

                pred_x0 = None
                pred_eps = None

                for u in range(resample_u):
                    pred_x0, pred_eps = self._ddim_predict_x0_eps(
                        img,
                        time,
                        cond,
                        precip_mask,
                        precip_visible=precip_visible,
                    )

                    if u < resample_u - 1:
                        alpha_t = self.alphas_cumprod[time]
                        eps = torch.randn_like(img)
                        img = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(1.0 - alpha_t) * eps

                if time_next < 0:
                    img = pred_x0
                else:
                    alpha_next = self.alphas_cumprod[time_next]
                    img = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * pred_eps

                self._maybe_append_img(imgs, img)

            return self._finalize_imgs(imgs, img)

        times = self._ddim_timesteps(total_steps)
        times_next = times[1:] + [-1]

        for time, time_next in zip(times, times_next):
            img, _, _ = self._ddim_step(
                img,
                time,
                time_next,
                cond,
                precip_mask,
                eta,
                precip_visible=precip_visible,
            )
            self._maybe_append_img(imgs, img)

        return self._finalize_imgs(imgs, img)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
````

## What changed from your original file

### Architectural changes
- Added:
  - `self.visible_precip_adapter`
  - `self.visible_precip_modality_emb`
- `_condition_tokens()` now accepts:
  - `precip_visible`
- `forward()` and all relevant internal forwards now accept:
  - `precip_visible`

### Strict DiffMAE sampling choice
- Removed `_apply_visible_precip()` usage from:
  - `ddpm`
  - `ddim`
  - `repaint`
  - `repaint_ddim`

So visible precip is **not** directly pasted into the sampled image during inference.

### Memory fix
- `imgs` is only created if `return_all_timesteps=True`

### DDIM robustness
- `torch.linspace(...).int()` schedule replaced by `_ddim_timesteps()`
- square root terms are clamped for stability

## Final recommendation
I think this is the cleanest “strict DiffMAE” version of your current framework.

The only thing I’d still call out is training/inference consistency:
- to fully benefit from this, you should make sure **training actually provides `precip_visible` as conditioning input** whenever visible precip exists
- otherwise the model won’t learn to use this new conditioning path

If you want, I can next give you:
1. a **minimal diff patch** against your current file, or  
2. a **training-call update checklist** showing everywhere `p_losses()` / `sample_precip()` now need `precip_visible`.

## @huahuou

a training-call update checklist showing everywhere p_losses() / sample_precip() now need precip_visible

## Copilot

Yes — here’s the **training-call update checklist** for the strict DiffMAE version.

## Direct answer
Anywhere you currently call:

- `model.p_losses(...)`
- `model.sample_precip(...)`
- optionally `model.forward(...)`, `model_predictions(...)`, or `p_mean_variance(...)`

you now need to decide whether to pass:

```python
precip_visible=...
```

If you want the new clean-visible-conditioning path to actually work, **training and inference must both supply `precip_visible` consistently**.

---

# 1. New calling convention

## `p_losses`
Old:
```python
loss_dict = model.p_losses(
    x_start=precip_target,
    cond=cond,
    precip_mask=precip_mask,
)
```

New:
```python
loss_dict = model.p_losses(
    x_start=precip_target,
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
)
```

## `sample_precip`
Old:
```python
sample = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    sampler="ddim",
)
```

New:
```python
sample = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
    sampler="ddim",
)
```

---

# 2. What `precip_visible` should be

For the strict DiffMAE version:

- `x_start`: the **full clean precip target**
- `precip_mask`: indicates which regions are masked / to be predicted
- `precip_visible`: the **clean visible subset of precip**, with the masked region removed/zeroed

So typically:

```python
pixel_mask = model.expand_patch_mask(precip_mask, H, W).to(x_start.dtype)
precip_visible = x_start * (1.0 - pixel_mask)
```

Where:
- `pixel_mask == 1` means masked / to predict
- `pixel_mask == 0` means visible / known

That matches your current loss masking semantics.

---

# 3. Training checklist

## A. Dataset / batch construction
You need each training batch to provide enough information to build:

- `x_start` = full precip target
- `cond` = other modalities
- `precip_mask`
- `precip_visible`

### Checklist
- [ ] Full precip target tensor is available
- [ ] Mask tensor is available at patch level or can be generated
- [ ] Visible precip tensor is computed from the full target and mask
- [ ] Visible precip uses the same normalization as `x_start`

### Recommended construction
If you already have full target `precip` and a patch mask:

```python
pixel_mask = model.expand_patch_mask(precip_mask, precip.shape[-2], precip.shape[-1]).to(precip.dtype)
precip_visible = precip * (1.0 - pixel_mask)
```

If grouped precip is used, `expand_patch_mask()` already handles grouped layouts.

---

## B. Training step
Wherever your trainer computes the diffusion loss:

### Before
```python
loss_dict = model.p_losses(
    x_start=precip,
    cond=cond,
    precip_mask=precip_mask,
)
```

### After
```python
loss_dict = model.p_losses(
    x_start=precip,
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
)
```

### Checklist
- [ ] `precip_visible` is passed into every training loss call
- [ ] same mask used to build `precip_visible` is passed as `precip_mask`
- [ ] no mismatch between visible region and loss mask

---

## C. Validation loss
If you compute validation diffusion loss with `p_losses`, update that too.

### Checklist
- [ ] validation `p_losses(...)` call passes `precip_visible`
- [ ] validation masking logic matches training masking logic

---

# 4. Sampling / inference checklist

## A. Inpainting inference
At inference time, if you want the model to use known visible precip as conditioning, you must pass it in.

### Before
```python
pred = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    sampler="repaint_ddim",
)
```

### After
```python
pred = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
    sampler="repaint_ddim",
)
```

### Checklist
- [ ] known visible precip is supplied at inference
- [ ] masked region in `precip_visible` is zeroed out
- [ ] visible precip normalization matches training normalization
- [ ] `precip_mask` is exactly the same mask that defines visible vs masked target area

---

## B. Fully unconditional target generation
If you want to generate precip without any visible precip context, you can still omit it:

```python
pred = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=None,
    sampler="ddim",
)
```

But then the new visible-conditioning path is unused.

### Checklist
- [ ] decide explicitly whether inference is “with visible precip conditioning” or “without it”
- [ ] don’t accidentally pass `None` during evaluation if training always used visible precip

---

# 5. Internal/helper calls that may also need updating

You may not call these directly, but if you do, they now accept `precip_visible` too:

- `forward(...)`
- `model_predictions(...)`
- `p_mean_variance(...)`

## `forward`
Old:
```python
out = model(noisy_precip, t, cond, precip_mask)
```

New:
```python
out = model(noisy_precip, t, cond, precip_mask, precip_visible=precip_visible)
```

## `model_predictions`
Old:
```python
pred_noise, pred_x0 = model.model_predictions(x_t, t, cond, precip_mask)
```

New:
```python
pred_noise, pred_x0 = model.model_predictions(
    x_t, t, cond, precip_mask, precip_visible=precip_visible
)
```

## `p_mean_variance`
Old:
```python
model_mean, posterior_variance, posterior_log_variance, x0 = model.p_mean_variance(
    x_t, t, cond, precip_mask
)
```

New:
```python
model_mean, posterior_variance, posterior_log_variance, x0 = model.p_mean_variance(
    x_t, t, cond, precip_mask, precip_visible=precip_visible
)
```

### Checklist
- [ ] grep for direct calls to `forward(`
- [ ] grep for `model_predictions(`
- [ ] grep for `p_mean_variance(`
- [ ] add `precip_visible=` where needed

---

# 6. Repository-wide grep checklist

Search for these patterns in your repo:

```python name=search_targets.txt
p_losses(
sample_precip(
model_predictions(
p_mean_variance(
forward(
```

Then inspect each occurrence.

## High-priority updates
Most likely places:
- training loop / trainer
- validation loop
- sampling/eval notebooks
- any standalone inference scripts
- any diagnostic scripts that call diffusion internals directly

---

# 7. Data contract checklist

To avoid silent bugs, ensure:

- [ ] `precip_visible.shape == x_start.shape`
- [ ] `precip_visible.dtype == x_start.dtype`
- [ ] `precip_visible.device == x_start.device`
- [ ] masked region of `precip_visible` is zero
- [ ] visible region of `precip_visible` contains clean normalized precip
- [ ] `precip_mask` semantics are unchanged: 1 = masked, 0 = visible

Recommended assertion:

```python
pixel_mask = model.expand_patch_mask(precip_mask, x_start.shape[-2], x_start.shape[-1]).to(x_start.dtype)
assert precip_visible.shape == x_start.shape
assert torch.allclose(precip_visible * pixel_mask, torch.zeros_like(precip_visible), atol=1e-6)
```

---

# 8. Training-consistency checklist

This is the most important conceptual part.

If you train with visible precip conditioning, then inference should also provide visible precip conditioning.

If you train **without** passing `precip_visible`, then adding it only at sampling time will not help much because the model never learned to use that path.

### Choose one regime explicitly

## Regime A: strict DiffMAE inpainting
- Training: pass `precip_visible`
- Validation: pass `precip_visible`
- Sampling: pass `precip_visible`

## Regime B: no visible-precip conditioning
- Training: `precip_visible=None`
- Validation: `precip_visible=None`
- Sampling: `precip_visible=None`

For your goal, you want **Regime A**.

---

# 9. Minimal example training step

```python name=training_step_example.py
precip = batch["precip"]                     # (B, C, H, W), full clean target
cond = {
    "background": batch["background"],
    "surface": batch["surface"],
    "forcing": batch["forcing"],
    "reflectivity": batch["reflectivity"],
}
precip_mask = batch["precip_mask"]           # patch mask, semantics: 1=masked, 0=visible

pixel_mask = model.expand_patch_mask(
    precip_mask, precip.shape[-2], precip.shape[-1]
).to(precip.dtype)

precip_visible = precip * (1.0 - pixel_mask)

loss_dict = model.p_losses(
    x_start=precip,
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
)

loss = loss_dict["loss"]
loss.backward()
```

---

# 10. Minimal example sampling step

```python name=sampling_example.py
cond = {
    "background": batch["background"],
    "surface": batch["surface"],
    "forcing": batch["forcing"],
    "reflectivity": batch["reflectivity"],
}
precip = batch["precip"]
precip_mask = batch["precip_mask"]

pixel_mask = model.expand_patch_mask(
    precip_mask, precip.shape[-2], precip.shape[-1]
).to(precip.dtype)

precip_visible = precip * (1.0 - pixel_mask)

pred = model.sample_precip(
    cond=cond,
    precip_mask=precip_mask,
    precip_visible=precip_visible,
    sampler="ddim",
    return_all_timesteps=False,
)
```

---

# 11. Common mistakes to avoid

- [ ] Passing full precip as `precip_visible` without masking out hidden regions
- [ ] Using a different mask to build `precip_visible` than the one passed as `precip_mask`
- [ ] Training without `precip_visible`, then expecting conditioned sampling to work
- [ ] Forgetting validation/inference updates after changing training
- [ ] Supplying unnormalized visible precip while target/conditions are normalized

---

# 12. Best practical next step
Search your repo for the training and sampling call sites, then patch them in this order:

1. training loss call
2. validation loss call
3. inference / notebook sampling call
4. any direct calls to `model_predictions` or `p_mean_variance`

If you want, I can next help with one of these:
1. **generate a grep-based patch checklist for your repo paths**, or  
2. **review your trainer script if you paste it here**.
