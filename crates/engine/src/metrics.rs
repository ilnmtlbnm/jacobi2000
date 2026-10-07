//! The subset of `train::metrics` that `crates/web` needs at runtime
//! (correlation, peak, bias, radial profile, cross-correlation lag, h
//! decode), ported verbatim (same formulas) so the demo page's live
//! comparison metrics don't pull Burn in through `train`. The full set
//! (RMSE, banded errors, rollout scoring, round-trip scoring) stays in
//! `train::metrics` for the CLI's training-time eval tools.

use crate::norm::Normalizer;

/// Pearson correlation. A constant field has no variance to correlate and
/// scores 0 rather than a division by zero.
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().map(|&x| x as f64).sum::<f64>() / n, b.iter().map(|&x| x as f64).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa * sbb).sqrt()) as f32
}

pub fn peak(f: &[f32]) -> f32 {
    f.iter().fold(0.0f32, |a, &x| a.max(x.abs()))
}

pub fn mean(f: &[f32]) -> f32 {
    f.iter().sum::<f32>() / f.len() as f32
}

pub fn std_dev(f: &[f32]) -> f32 {
    let m = mean(f);
    (f.iter().map(|x| (x - m).powi(2)).sum::<f32>() / f.len() as f32).sqrt()
}

/// `(mean(model) - mean(solver)) / std(solver)`.
pub fn bias(model: &[f32], solver: &[f32]) -> f32 {
    (mean(model) - mean(solver)) / std_dev(solver).max(1e-12)
}

/// Azimuthally-averaged `|f|` profile about `(cx, cy)`, one value per
/// one-cell-wide radius bin, cut off at the nearest wall.
pub fn radial_profile(f: &[f32], nx: usize, ny: usize, cx: f32, cy: f32) -> Vec<f64> {
    let rmax = cx.min(cy).min(nx as f32 - cx).min(ny as f32 - cy).floor().max(1.0) as usize;
    let mut sum = vec![0.0f64; rmax + 1];
    let mut count = vec![0u32; rmax + 1];
    for y in 0..ny {
        for x in 0..nx {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt();
            let bin = (r + 0.5) as usize;
            if bin <= rmax {
                sum[bin] += f[y * nx + x].abs() as f64;
                count[bin] += 1;
            }
        }
    }
    sum.iter().zip(&count).map(|(s, &c)| if c == 0 { 0.0 } else { s / c as f64 }).collect()
}

fn corr_at_lag(model: &[f64], solver: &[f64], lag: isize) -> f32 {
    let n = model.len() as isize;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for i in 0..n {
        let si = i - lag;
        if si >= 0 && si < n {
            a.push(model[i as usize] as f32);
            b.push(solver[si as usize] as f32);
        }
    }
    if a.len() < 3 {
        return f32::NEG_INFINITY;
    }
    correlation(&a, &b)
}

/// Lag (in cells) that maximizes the cross-correlation of the model's radial
/// profile against the solver's, refined to sub-cell accuracy.
pub fn xcorr_lag(model: &[f64], solver: &[f64], max_lag: usize) -> f32 {
    let max_lag = max_lag as isize;
    let (mut best_lag, mut best_c) = (0isize, f32::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let c = corr_at_lag(model, solver, lag);
        if c > best_c {
            best_c = c;
            best_lag = lag;
        }
    }
    if best_lag > -max_lag && best_lag < max_lag {
        let (l, c, r) = (corr_at_lag(model, solver, best_lag - 1), best_c, corr_at_lag(model, solver, best_lag + 1));
        if l.is_finite() && r.is_finite() {
            let den = l - 2.0 * c + r;
            if den < 0.0 {
                return best_lag as f32 + 0.5 * (l - r) / den;
            }
        }
    }
    best_lag as f32
}

/// Decode the packed `h` channel (the first `n` of `x`) back to physical
/// units.
pub fn decode_h(x: &[f32], norm: &Normalizer, n: usize) -> Vec<f32> {
    x[..n].iter().map(|v| v * norm.std[0] + norm.mean[0]).collect()
}

pub fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let s: f64 = a.iter().zip(b).map(|(x, y)| { let d = (x - y) as f64; d * d }).sum();
    (s / a.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_of_identical_fields_is_one() {
        let a = [0.1f32, 0.5, -0.3, 0.9, -0.7];
        assert!((correlation(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn correlation_of_a_constant_field_is_zero() {
        let a = [1.0f32; 5];
        let b = [0.1f32, 0.5, -0.3, 0.9, -0.7];
        assert_eq!(correlation(&a, &b), 0.0);
    }

    #[test]
    fn bias_of_identical_fields_is_zero() {
        let a = [0.1f32, 0.5, -0.3, 0.9, -0.7];
        assert_eq!(bias(&a, &a), 0.0);
    }
}
