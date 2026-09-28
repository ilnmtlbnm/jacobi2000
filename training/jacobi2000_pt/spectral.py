"""Mirrors crates/train/src/spectral.rs: a Laplacian-pyramid band-MSE term,
not an FFT. Three levels, short wavelengths weighted most."""

import torch
import torch.nn.functional as F

BAND_WEIGHTS = [1.0, 0.5, 0.25]


def band_mse(err: torch.Tensor) -> torch.Tensor:
    low = err
    loss = torch.zeros((), device=err.device, dtype=err.dtype)
    for w in BAND_WEIGHTS:
        nxt = F.avg_pool2d(low, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        band = low - nxt
        loss = loss + band.pow(2).mean() * w
        low = nxt
    return loss
