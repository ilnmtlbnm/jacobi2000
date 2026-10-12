"""The Jacobi2000 patch transformer in PyTorch. `crates/engine` implements the
same forward pass for inference.

Every layer is named and shaped to match the exchange format
(see docs/pytorch-training.md), so `export.py` writes safetensors the Rust
side reads with no reshuffling. "Linear" layers here are hand-rolled
(weight stored `[in, out]`, `y = x @ W + b`) rather than `nn.Linear`, whose
PyTorch-native `[out, in]` layout would need a transpose on every export.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Jacobi2000Config:
    nx: int = 64
    ny: int = 64
    patch: int = 4
    channels: int = 3
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ff_mult: int = 4
    meta_dim: int = 5
    head_dim: int = 32

    @property
    def tokens_x(self) -> int:
        return self.nx // self.patch

    @property
    def tokens_y(self) -> int:
        return self.ny // self.patch

    @property
    def tokens(self) -> int:
        return self.tokens_x * self.tokens_y

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch * self.patch


class Lin(nn.Module):
    """`y = x @ weight + bias`, `weight: [in, out]` (row layout, not
    PyTorch's `nn.Linear` `[out, in]`)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        k = 1.0 / (d_in ** 0.5)
        self.weight = nn.Parameter(torch.empty(d_in, d_out).uniform_(-k, k))
        self.bias = nn.Parameter(torch.empty(d_out).uniform_(-k, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class Norm(nn.Module):
    """LayerNorm: normalize over the last dim, scale by
    `gamma`, shift by `beta`, `eps = 1e-5`."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.gamma, self.beta, eps=1e-5)


class Conv(nn.Module):
    """`weight: [out, in, kh, kw]`, PyTorch's own `nn.Conv2d` layout, which
    this wraps to keep a plain weight/bias pair for export."""

    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        fan_in = cin * k * k
        bound = (1.0 / fan_in) ** 0.5
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(cout).uniform_(-bound, bound))
        self.padding = k // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=self.padding)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.heads = heads
        self.norm1 = Norm(dim)
        self.q = Lin(dim, dim)
        self.k = Lin(dim, dim)
        self.v = Lin(dim, dim)
        self.proj = Lin(dim, dim)
        self.norm2 = Norm(dim)
        self.ff1 = Lin(dim, dim * ff_mult)
        self.ff2 = Lin(dim * ff_mult, dim)

    def attention(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, t, d = x.shape
        hd = d // self.heads
        split = lambda y: y.reshape(b, t, self.heads, hd).permute(0, 2, 1, 3)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / (hd ** 0.5)
        if mask is not None:
            scores = scores + mask
        w = torch.softmax(scores, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).reshape(b, t, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.proj(self.attention(self.norm1(x), mask))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


def upsample(x: torch.Tensor, dim: int, p: int) -> torch.Tensor:
    """Linear upsample of spatial `dim` (2 = y, 3 = x) by `p`, replicating
    the last token so the far edge has a neighbour to interpolate towards.
    `engine`'s `upsample` does the same."""
    n = x.shape[dim]
    nxt = torch.cat(
        [x.narrow(dim, 1, n - 1), x.narrow(dim, n - 1, 1)], dim=dim
    )
    rep = x.repeat_interleave(p, dim=dim)
    rep_next = nxt.repeat_interleave(p, dim=dim)
    w = ((torch.arange(n * p, device=x.device, dtype=x.dtype) % p) + 0.5) / p
    shape = [1, 1, 1, 1]
    shape[dim] = n * p
    w = w.reshape(shape)
    return rep * (1.0 - w) + rep_next * w


class CellHead(nn.Module):
    def __init__(self, dim: int, head_dim: int, channels: int):
        super().__init__()
        self.token_mix = Conv(dim, dim, 3)
        self.to_cells = Conv(dim, head_dim, 1)
        self.conv1 = Conv(head_dim + channels, head_dim, 3)
        self.conv2 = Conv(head_dim, channels, 3)

    def forward(self, x: torch.Tensor, state: torch.Tensor, ty: int, tx: int, patch: int) -> torch.Tensor:
        b, _, d = x.shape
        t = x.reshape(b, ty, tx, d).permute(0, 3, 1, 2)
        t = t + F.gelu(self.token_mix(t))
        z = self.to_cells(t)
        z = upsample(z, 2, patch)
        z = upsample(z, 3, patch)
        z = torch.cat([z, state], dim=1)
        return self.conv2(F.gelu(self.conv1(z)))


class Jacobi2000(nn.Module):
    def __init__(self, cfg: Jacobi2000Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Lin(cfg.patch_dim, cfg.dim)
        self.meta = Lin(cfg.meta_dim, cfg.dim)
        # [1, tokens, dim]: rank 3 (not 2), the shape the
        # safetensors exchange format stores.
        self.pos = nn.Parameter(torch.zeros(1, cfg.tokens, cfg.dim))
        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.ff_mult) for _ in range(cfg.layers)]
        )
        self.norm = Norm(cfg.dim)
        self.head = CellHead(cfg.dim, cfg.head_dim, cfg.channels)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, c = x.shape[0], x.shape[1]
        p, tx, ty = cfg.patch, cfg.tokens_x, cfg.tokens_y
        return (
            x.reshape(b, c, ty, p, tx, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, cfg.tokens, cfg.patch_dim)
        )

    def forward(
        self,
        state: torch.Tensor,
        meta: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """`state`: `[batch, channels, ny, nx]`. `meta`: `[batch, meta_dim]`.
        `mask`: additive `[tokens, tokens]` or `None`. Returns the predicted
        increment, same shape as `state`."""
        cfg = self.cfg
        x = self.patchify(state)
        x = self.embed(x) + self.pos
        m = self.meta(meta).unsqueeze(1)
        x = x + m

        mask4 = mask.reshape(1, 1, cfg.tokens, cfg.tokens) if mask is not None else None
        for block in self.blocks:
            x = block(x, mask4)

        return self.head(self.norm(x), state, cfg.tokens_y, cfg.tokens_x, cfg.patch)

    def step(
        self,
        state: torch.Tensor,
        meta: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return state + self.forward(state, meta, mask)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def stencil_mask(cfg: Jacobi2000Config, radius: int, dilation: int, device=None) -> torch.Tensor:
    """Additive `[tokens, tokens]` mask, the same
    neighbourhood rule as `engine::Mask::Stencil` (`-inf` where a key is not
    attended)."""
    n = cfg.tokens
    tx = cfg.tokens_x
    data = torch.full((n, n), float("-inf"))
    for q in range(n):
        qx, qy = q % tx, q // tx
        for k in range(n):
            kx, ky = k % tx, k // tx
            dx, dy = kx - qx, ky - qy
            if dx % dilation == 0 and dy % dilation == 0 \
                    and abs(dx // dilation) <= radius and abs(dy // dilation) <= radius:
                data[q, k] = 0.0
    return data.to(device) if device is not None else data
