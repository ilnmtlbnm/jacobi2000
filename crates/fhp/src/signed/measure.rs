#![allow(clippy::too_many_arguments)]
//! The 3A measurements (shear-wave decay, transverse-wave advection, rest
//! noise) for the signed engine, with the same setups and fits
//! (`ternary::measure`), on the represented (float) fields.

use super::{Params, Signed};
pub use crate::ternary::measure::{fit_line, phase_speed, position, shear_nu, Series};

/// Complex projection of u_x (or u_y) on exp(-i k . r), normalized as
/// `ternary::measure`'s: the amplitude of u = A sin(k.r + phi).
pub fn project(l: &Signed, comp_y: bool, kx: f64, ky: f64) -> (f64, f64) {
    let (m, jx, jy) = l.fields();
    let (w, h) = (l.w, l.h);
    let (mut re, mut im, mut mt) = (0.0, 0.0, 0.0);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (px, py) = position(x, y);
            let ph = kx * px + ky * py;
            let v = if comp_y { jy[i] } else { jx[i] };
            re += v * ph.cos();
            im -= v * ph.sin();
            mt += m[i];
        }
    }
    let n = mt / (w * h) as f64;
    (2.0 * re / (n * (w * h) as f64), 2.0 * im / (n * (w * h) as f64))
}

pub fn lattice(w: usize, h: usize, p: Params, seed: u32, u: impl Fn(f64, f64) -> (f64, f64)) -> Signed {
    let rho0 = 7.0 * p.n0;
    Signed::sample(
        w,
        h,
        Params { seed, ..p },
        |_, _| false,
        |x, y| {
            let v = u(x, y);
            (rho0, v.0, v.1)
        },
        seed,
    )
}

/// Shear wave u_x = u0 sin(k y), k = 2 pi m / (h sqrt3/2).
pub fn shear_series(p: Params, w: usize, h: usize, m: usize, u0: f64, seed: u32, steps: usize, every: usize) -> Series {
    let ky = 2.0 * std::f64::consts::PI * m as f64 / (h as f64 * 3f64.sqrt() / 2.0);
    let mut l = lattice(w, h, p, seed, |_, y| (u0 * (ky * y).sin(), 0.0));
    let mut s = Series { t: vec![], re: vec![], im: vec![] };
    let mut t = 0;
    loop {
        let (re, im) = project(&l, false, 0.0, ky);
        s.t.push(t as f64);
        s.re.push(re);
        s.im.push(im);
        if t >= steps {
            break;
        }
        l.run(every);
        t += every;
    }
    s
}

/// Transverse wave u_y = a0 sin(k x) in a uniform flow u_x = U; returns the
/// series and the measured mean u_x.
pub fn advect_series(p: Params, w: usize, h: usize, u: f64, m: usize, a0: f64, seed: u32, steps: usize, every: usize) -> (Series, f64) {
    let kx = 2.0 * std::f64::consts::PI * m as f64 / w as f64;
    let mut l = lattice(w, h, p, seed, |x, _| (u, a0 * (kx * x).sin()));
    let (ms, jx, _) = l.fields();
    let ux = jx.iter().sum::<f64>() / ms.iter().sum::<f64>();
    let mut s = Series { t: vec![], re: vec![], im: vec![] };
    let mut t = 0;
    loop {
        let (re, im) = project(&l, true, kx, 0.0);
        s.t.push(t as f64);
        s.re.push(re);
        s.im.push(im);
        if t >= steps {
            break;
        }
        l.run(every);
        t += every;
    }
    (s, ux)
}

/// Standard deviation of block-averaged u_x and u_y at rest, as
/// `ternary::measure::rest_noise`.
pub fn rest_noise(p: Params, w: usize, h: usize, seed: u32, blocks: &[usize], warm: usize, snaps: usize, gap: usize) -> Vec<(usize, f64, f64)> {
    let mut l = lattice(w, h, p, seed, |_, _| (0.0, 0.0));
    l.run(warm);
    let mut acc = vec![(0.0f64, 0.0f64, 0usize); blocks.len()];
    for _ in 0..snaps {
        let (m, jx, jy) = l.fields();
        for (bi, &b) in blocks.iter().enumerate() {
            let (nx, ny) = (w / b, h / b);
            let mut sums = vec![(0.0, 0.0, 0.0); nx * ny];
            for y in 0..ny * b {
                for x in 0..nx * b {
                    let i = y * w + x;
                    let s = &mut sums[(y / b) * nx + x / b];
                    s.0 += jx[i];
                    s.1 += jy[i];
                    s.2 += m[i];
                }
            }
            for s in &sums {
                let (ux, uy) = (s.0 / s.2, s.1 / s.2);
                acc[bi].0 += ux * ux;
                acc[bi].1 += uy * uy;
                acc[bi].2 += 1;
            }
        }
        l.run(gap);
    }
    blocks
        .iter()
        .zip(acc)
        .map(|(&b, a)| (b, (a.0 / a.2 as f64).sqrt(), (a.1 / a.2 as f64).sqrt()))
        .collect()
}

/// Independent half-filled deviations (variance 2/3 per channel):
/// std = sqrt(3 (2/3) / b^2) / (7 n0).
pub fn noise_theory(n0: f64, b: usize) -> f64 {
    (2.0 / (b * b) as f64).sqrt() / (7.0 * n0)
}

/// Orientation of the structures of a periodic coarse field `o` (`nx` x
/// `ny` display blocks, rows sqrt(3)/2 as tall as columns are wide), as the
/// band diagnosis's `tilt.py`: smoothed with a periodic Gaussian of `sigma`
/// blocks (y scaled as there), then the structure tensor J = <grad o grad
/// o^T> over columns [x0, x1). Returns tilt = Jxy / sqrt(Jxx Jyy) (< 0 for
/// "/" structures), aspect sqrt(l_max / l_min), and the long axis's angle
/// from +x in degrees, y up.
pub fn orientation(o: &[f32], nx: usize, ny: usize, sigma: f64, x0: usize, x1: usize) -> (f64, f64, f64) {
    let kernel = |s: f64, n: usize| -> Vec<f64> {
        let r = ((4.0 * s).ceil() as usize).min(n / 2);
        let k: Vec<f64> = (0..=2 * r).map(|i| (-((i as f64 - r as f64).powi(2)) / (2.0 * s * s)).exp()).collect();
        let t: f64 = k.iter().sum();
        k.iter().map(|v| v / t).collect()
    };
    let (kx, ky) = (kernel(sigma, nx), kernel(sigma * 3f64.sqrt() / 2.0, ny));
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    let mut a = vec![0.0f64; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            a[j * nx + i] = kx.iter().enumerate().map(|(t, w)| w * o[j * nx + (i + nx + t - rx) % nx] as f64).sum();
        }
    }
    let mut s = vec![0.0f64; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            s[j * nx + i] = ky.iter().enumerate().map(|(t, w)| w * a[((j + ny + t - ry) % ny) * nx + i]).sum();
        }
    }
    let (mut jxx, mut jxy, mut jyy, mut n) = (0.0, 0.0, 0.0, 0.0);
    let dy = 3f64.sqrt() / 2.0;
    for j in 0..ny {
        for i in x0..x1 {
            let gx = (s[j * nx + (i + 1) % nx] - s[j * nx + (i + nx - 1) % nx]) / 2.0;
            let gy = (s[((j + 1) % ny) * nx + i] - s[((j + ny - 1) % ny) * nx + i]) / 2.0 / dy;
            jxx += gx * gx;
            jxy += gx * gy;
            jyy += gy * gy;
            n += 1.0;
        }
    }
    let (jxx, jxy, jyy) = (jxx / n, jxy / n, jyy / n);
    let m = (jxx + jyy) / 2.0;
    let d = (((jxx - jyy) / 2.0).powi(2) + jxy * jxy).sqrt();
    let aspect = ((m + d) / (m - d).max(1e-300)).sqrt();
    let angle = (0.5 * (2.0 * jxy).atan2(jxx - jyy)).to_degrees() + 90.0;
    (jxy / (jxx * jyy).sqrt(), aspect, angle.rem_euclid(180.0))
}

/// Mirror-odd statistics of a lattice's coarse field (blocks of `db`
/// sites): the velocity correlation <u_x' u_y'> / (rms u_x' rms u_y'), the
/// vorticity tilt (`orientation`, sigma 1 block, whole width), and the
/// vorticity skewness <w^3> / <w^2>^(3/2). Each changes sign under a
/// y-mirror of the flow.
pub fn mirror_odd(l: &Signed, db: usize) -> [f64; 3] {
    let mut f = crate::coarse::Field::new(l.w / db, l.h / db, db, db);
    f.add(&l.page_sums(db, db), 1.0);
    let n = f.ux.len() as f64;
    let (mx, my) = (f.ux.iter().map(|&v| v as f64).sum::<f64>() / n, f.uy.iter().map(|&v| v as f64).sum::<f64>() / n);
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (a, b) in f.ux.iter().zip(&f.uy) {
        let (a, b) = (*a as f64 - mx, *b as f64 - my);
        xy += a * b;
        xx += a * a;
        yy += b * b;
    }
    let o = f.vorticity();
    let (tilt, _, _) = orientation(&o, f.nx, f.ny, 1.0, 0, f.nx);
    let m2 = o.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    let m3 = o.iter().map(|&v| (v as f64).powi(3)).sum::<f64>() / n;
    [xy / (xx * yy).sqrt(), tilt, m3 / m2.powf(1.5)]
}

/// A Taylor-Green array (amplitude a, one cell pair per period) in a
/// uniform flow u, plus a shear wave of amplitude e along the "/" diagonal,
/// so the flow has no mirror symmetry of its own; `mirror` gives its image
/// under y -> -y (u_y -> -u_y). Periods: the lattice (w, h sqrt3/2).
pub fn handed_field(w: usize, h: usize, u: f64, a: f64, e: f64, mirror: bool) -> impl Fn(f64, f64) -> (f64, f64) {
    let tau = 2.0 * std::f64::consts::PI;
    let (kx, ky) = (tau / w as f64, tau / (h as f64 * 3f64.sqrt() / 2.0));
    let sgn = if mirror { -1.0 } else { 1.0 };
    let kn = (kx * kx + ky * ky).sqrt();
    move |px, py| {
        let py = sgn * py;
        let tg = (a * (kx * px).sin() * (ky * py).cos(), -a * (kx / ky) * (kx * px).cos() * (ky * py).sin());
        let ph = (kx * px + ky * py).sin();
        let sw = (-e * ky / kn * ph, e * kx / kn * ph);
        (u + tg.0 + sw.0, sgn * (tg.1 + sw.1))
    }
}

/// Mirror test: per seed, the lattice sampled from `handed_field` and from
/// its mirror image, both run `steps` steps; returns per seed the
/// `mirror_odd` statistics of the two. A mirror-symmetric scheme gives
/// statistics of opposite signs on average; their half sum is the bias.
pub fn mirror_pairs(p: Params, w: usize, h: usize, u: f64, a: f64, e: f64, seeds: &[u32], steps: usize, db: usize) -> Vec<([f64; 3], [f64; 3])> {
    seeds
        .iter()
        .map(|&seed| {
            let run = |mirror: bool| {
                let mut l = lattice(w, h, p, seed, handed_field(w, h, u, a, e, mirror));
                l.run(steps);
                mirror_odd(&l, db)
            };
            (run(false), run(true))
        })
        .collect()
}
