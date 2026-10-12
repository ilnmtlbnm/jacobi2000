"""Scoring functions (the same formulas as `crates/engine/src/metrics.rs`) for
fast in-Python iteration. The acceptance numbers come from `eval-metrics`
(crates/engine-cli) on the exported checkpoint; this exists so training
doesn't need a Rust round trip to see whether it's working."""

from dataclasses import dataclass

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    ma, mb = a.mean(), b.mean()
    da, db = a - ma, b - mb
    saa, sbb = (da * da).sum(), (db * db).sum()
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    return float((da * db).sum() / np.sqrt(saa * sbb))


def peak(f: np.ndarray) -> float:
    return float(np.max(np.abs(f)))


def mean(f: np.ndarray) -> float:
    return float(np.mean(f))


def std_dev(f: np.ndarray) -> float:
    return float(np.std(f))


def bias(model: np.ndarray, solver: np.ndarray) -> float:
    return (mean(model) - mean(solver)) / max(std_dev(solver), 1e-12)


def active_mask(truth: np.ndarray) -> np.ndarray:
    return np.abs(truth) > 0.1 * peak(truth)


def masked_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return rmse(pred[mask], truth[mask])


def radial_profile(f: np.ndarray, nx: int, ny: int, cx: float, cy: float) -> np.ndarray:
    rmax = max(int(np.floor(min(cx, cy, nx - cx, ny - cy))), 1)
    ys, xs = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    bins = (r + 0.5).astype(np.int64)
    sums = np.zeros(rmax + 1, dtype=np.float64)
    counts = np.zeros(rmax + 1, dtype=np.int64)
    valid = bins <= rmax
    np.add.at(sums, bins[valid], np.abs(f).reshape(ny, nx)[valid])
    np.add.at(counts, bins[valid], 1)
    out = np.zeros(rmax + 1, dtype=np.float64)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def profile_peak(prof: np.ndarray) -> float:
    k = int(np.argmax(prof))
    if k == 0 or k + 1 >= len(prof):
        return float(k)
    left, center, right = prof[k - 1], prof[k], prof[k + 1]
    den = left - 2 * center + right
    if den >= 0.0:
        return float(k)
    return k + 0.5 * (left - right) / den


def wavefront_radius(f: np.ndarray, nx: int, ny: int, cx: float, cy: float) -> float:
    return profile_peak(radial_profile(f, nx, ny, cx, cy))


def _corr_at_lag(model: np.ndarray, solver: np.ndarray, lag: int) -> float:
    n = len(model)
    idx = np.arange(n)
    si = idx - lag
    valid = (si >= 0) & (si < n)
    if valid.sum() < 3:
        return -np.inf
    return correlation(model[idx[valid]], solver[si[valid]])


def xcorr_lag(model: np.ndarray, solver: np.ndarray, max_lag: int) -> float:
    best_lag, best_c = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        c = _corr_at_lag(model, solver, lag)
        if c > best_c:
            best_c, best_lag = c, lag
    if -max_lag < best_lag < max_lag:
        left = _corr_at_lag(model, solver, best_lag - 1)
        right = _corr_at_lag(model, solver, best_lag + 1)
        if np.isfinite(left) and np.isfinite(right):
            den = left - 2 * best_c + right
            if den < 0.0:
                return best_lag + 0.5 * (left - right) / den
    return float(best_lag)


def decode_h(x: np.ndarray, mean0: float, std0: float, n: int) -> np.ndarray:
    return x[:n] * std0 + mean0


@dataclass
class Metrics:
    corr: float
    corr_h: float
    active_rmse: float
    amplitude_ratio: float
    xcorr_lag: float
    global_rmse: float
    bias: float


def score(pred: np.ndarray, truth: np.ndarray, mean0: float, std0: float, nx: int, ny: int, cx: float, cy: float) -> Metrics:
    n = nx * ny
    ph = decode_h(pred, mean0, std0, n)
    th = decode_h(truth, mean0, std0, n)
    mask = active_mask(th)
    mprof = radial_profile(ph, nx, ny, cx, cy)
    tprof = radial_profile(th, nx, ny, cx, cy)
    return Metrics(
        corr=correlation(pred, truth),
        corr_h=correlation(pred[:n], truth[:n]),
        active_rmse=masked_rmse(pred, truth, np.tile(mask, 3)),
        amplitude_ratio=peak(ph) / peak(th),
        xcorr_lag=xcorr_lag(mprof, tprof, 8),
        global_rmse=rmse(pred, truth),
        bias=bias(ph, th),
    )
