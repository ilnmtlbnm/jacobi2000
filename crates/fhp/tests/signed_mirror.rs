//! Whole-step mirror test of the signed engine: a flow without mirror
//! symmetry of its own (Taylor-Green array, uniform flow, a "/" shear wave)
//! and its y-mirror image, over seeds, 4000 steps at B16 N4. The mirror-odd
//! statistics (u_x u_y correlation, vorticity tilt, vorticity skewness) of
//! the two runs must be opposite on average: their half sum is the scheme's
//! handed bias. The random streams cannot be mirrored, so the test is
//! statistical over seeds (deterministic for fixed seeds).
//!
//! The first engine's column geometry (`COLUMNS`, which ignores that odd
//! rows sit half a site to the right) fails it with a "/" bias; the true
//! site positions pass. Its own test binary, since `COLUMNS` is global.

use fhp::signed::{measure, Params, COLUMNS};
use fhp::ternary::Rule;

/// Per statistic: mean half sum over seeds and its standard error.
fn bias(columns: bool) -> Vec<(f64, f64)> {
    COLUMNS.store(columns, std::sync::atomic::Ordering::Relaxed);
    let p = Params { seed: 1, rule: Rule::Optimal, block: 16, every: 4, n0: 4.0, boundary: None };
    let seeds: Vec<u32> = (1..=12).collect();
    let r = measure::mirror_pairs(p, 256, 128, 0.2, 0.1, 0.01, &seeds, 4000, 8);
    COLUMNS.store(false, std::sync::atomic::Ordering::Relaxed);
    (0..3)
        .map(|k| {
            let half: Vec<f64> = r.iter().map(|(x, y)| (x[k] + y[k]) / 2.0).collect();
            let n = half.len() as f64;
            let m = half.iter().sum::<f64>() / n;
            let se = (half.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
            (m, se)
        })
        .collect()
}

#[test]
fn mirror_image_evolves_as_mirror() {
    let names = ["u_x u_y correlation", "vorticity tilt", "vorticity skewness"];
    let old = bias(true);
    let new = bias(false);
    for (k, name) in names.iter().enumerate() {
        eprintln!("{name}: column geometry {:+.4} +- {:.4}, true positions {:+.4} +- {:.4}", old[k].0, old[k].1, new[k].0, new[k].1);
    }
    // The test sees the column geometry's "/" bias (tilt < 0, correlation > 0).
    assert!(old[0].0 > 5.0 * old[0].1 && old[1].0 < -5.0 * old[1].1, "column geometry: no bias seen {old:?}");
    for (k, name) in names.iter().enumerate() {
        assert!(new[k].0.abs() < 3.0 * new[k].1, "{name}: bias {:+.4} +- {:.4}", new[k].0, new[k].1);
    }
}
