//! RGBA images of a field (v in [0, 1]) or of its signed error against the
//! analytic solution, at most `MAX_IMAGE` pixels a side (nearest sample).

use crate::Problem;

pub const MAX_IMAGE: usize = 512;

pub fn image_size(n: usize) -> usize {
    n.min(MAX_IMAGE)
}

/// Viridis at 9 stops, linearly interpolated.
const VIRIDIS: [[f32; 3]; 9] = [
    [68.0, 1.0, 84.0],
    [71.0, 44.0, 122.0],
    [59.0, 81.0, 139.0],
    [44.0, 113.0, 142.0],
    [33.0, 144.0, 141.0],
    [39.0, 173.0, 129.0],
    [92.0, 200.0, 99.0],
    [170.0, 220.0, 50.0],
    [253.0, 231.0, 37.0],
];

fn viridis(t: f64) -> [u8; 3] {
    let t = (t.clamp(0.0, 1.0) * 8.0) as f32;
    let i = (t as usize).min(7);
    let f = t - i as f32;
    std::array::from_fn(|c| (VIRIDIS[i][c] * (1.0 - f) + VIRIDIS[i + 1][c] * f).round() as u8)
}

/// Signed error: white at |e| <= 1e-6, towards red above the analytic
/// value and blue below, saturating at |e| = 1, log scale in between.
fn error_colour(e: f64) -> [u8; 3] {
    let t = ((e.abs().max(1e-6).log10() + 6.0) / 6.0).clamp(0.0, 1.0);
    let fade = (255.0 * (1.0 - t)) as u8;
    if e >= 0.0 {
        [255, fade, fade]
    } else {
        [fade, fade, 255]
    }
}

/// `value(k)` is v at row-major index k of the n x n grid.
pub fn rgba(p: &Problem, value: impl Fn(usize) -> f64, error: bool) -> Vec<u8> {
    let n = p.n;
    let m = image_size(n);
    let mut out = Vec::with_capacity(m * m * 4);
    for a in 0..m {
        for c in 0..m {
            let (i, j) = (a * n / m, c * n / m);
            let v = value(i * n + j);
            let px = if error { error_colour(v - p.analytic(i, j)) } else { viridis(v) };
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
    out
}
