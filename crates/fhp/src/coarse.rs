//! Coarse-graining: block sums of mass and momentum, a running time
//! average, vorticity, and RGBA rendering.

/// Integer block sums, row-major over blocks: momentum in units
/// (2 px, 2 py / sqrt 3), particles, and fluid sites.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockSums {
    pub nx: usize,
    pub ny: usize,
    pub bw: usize,
    pub bh: usize,
    pub jx: Vec<i32>,
    pub jy: Vec<i32>,
    pub mass: Vec<i32>,
    pub fluid: Vec<i32>,
}

impl BlockSums {
    /// Add another sample of the same blocks.
    pub fn add(&mut self, o: &BlockSums) {
        for (a, b) in [(&mut self.jx, &o.jx), (&mut self.jy, &o.jy), (&mut self.mass, &o.mass), (&mut self.fluid, &o.fluid)] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

/// Block sums over bit-planes stored as u64 words, `wn` words per row:
/// planes 0..6 moving, 6 rest, and the obstacle mask. `bw` divides 64.
pub fn block_sums(w: usize, h: usize, planes: &[Vec<u64>; 7], obst: &[u64], bw: usize, bh: usize) -> BlockSums {
    assert!(64 % bw == 0 && w.is_multiple_of(bw) && h.is_multiple_of(bh));
    let wn = w / 64;
    let (nx, ny) = (w / bw, h / bh);
    let mut s = BlockSums { nx, ny, bw, bh, jx: vec![0; nx * ny], jy: vec![0; nx * ny], mass: vec![0; nx * ny], fluid: vec![0; nx * ny] };
    let per = 64 / bw;
    let mask = if bw == 64 { u64::MAX } else { (1u64 << bw) - 1 };
    for y in 0..h {
        let by = y / bh;
        for wi in 0..wn {
            let i = y * wn + wi;
            for k in 0..per {
                let sh = k * bw;
                let m = mask << sh;
                let pc = |p: &Vec<u64>| (p[i] & m).count_ones() as i32;
                let n: [i32; 7] = std::array::from_fn(|d| pc(&planes[d]));
                let b = by * nx + wi * per + k;
                s.jx[b] += 2 * n[0] + n[1] - n[2] - 2 * n[3] - n[4] + n[5];
                s.jy[b] += n[1] + n[2] - n[4] - n[5];
                s.mass[b] += n.iter().sum::<i32>();
                s.fluid[b] += bw as i32 - (obst[i] & m).count_ones() as i32;
            }
        }
    }
    s
}

/// Running time average of block velocity, and its vorticity. Blocks
/// holding any obstacle site are drawn solid.
pub struct Field {
    pub nx: usize,
    pub ny: usize,
    pub bw: usize,
    pub bh: usize,
    pub ux: Vec<f32>,
    pub uy: Vec<f32>,
    pub solid: Vec<bool>,
    pub samples: u32,
}

impl Field {
    pub fn new(nx: usize, ny: usize, bw: usize, bh: usize) -> Self {
        Field { nx, ny, bw, bh, ux: vec![0.0; nx * ny], uy: vec![0.0; nx * ny], solid: vec![false; nx * ny], samples: 0 }
    }

    /// Fold block sums into the average with weight `alpha` (the first
    /// sample replaces the field).
    pub fn add(&mut self, s: &BlockSums, alpha: f32) {
        let a = if self.samples == 0 { 1.0 } else { alpha };
        let sq3 = 3f32.sqrt();
        for b in 0..self.nx * self.ny {
            let m = s.mass[b].max(1) as f32;
            let ux = s.jx[b] as f32 * 0.5 / m;
            let uy = s.jy[b] as f32 * 0.5 * sq3 / m;
            self.ux[b] += a * (ux - self.ux[b]);
            self.uy[b] += a * (uy - self.uy[b]);
            self.solid[b] = s.fluid[b] < (s.bw * s.bh) as i32;
        }
        self.samples += 1;
    }

    /// Vorticity d uy/dx - d ux/dy by central differences, periodic,
    /// in units of 1/step (rows are sqrt(3)/2 apart).
    pub fn vorticity(&self) -> Vec<f32> {
        let (nx, ny) = (self.nx, self.ny);
        let dx = self.bw as f32;
        let dy = self.bh as f32 * 3f32.sqrt() / 2.0;
        let mut o = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let ip = j * nx + (i + 1) % nx;
                let im = j * nx + (i + nx - 1) % nx;
                let jp = ((j + 1) % ny) * nx + i;
                let jm = ((j + ny - 1) % ny) * nx + i;
                o[j * nx + i] = (self.uy[ip] - self.uy[im]) / (2.0 * dx) - (self.ux[jp] - self.ux[jm]) / (2.0 * dy);
            }
        }
        o
    }

    pub fn mean_ux(&self) -> f32 {
        let (mut s, mut n) = (0.0, 0);
        for b in 0..self.ux.len() {
            if !self.solid[b] {
                s += self.ux[b];
                n += 1;
            }
        }
        s / n.max(1) as f32
    }

    /// RGBA image, one pixel per block, top row = highest y: vorticity on a
    /// blue-white-red scale saturating at `vmax` (0 = 99th percentile of
    /// |vorticity|), obstacles black.
    pub fn vorticity_rgba(&self, vmax: f32) -> Vec<u8> {
        let (o, vmax) = self.scaled_vorticity(vmax);
        self.rgba(&o, vmax, diverging)
    }

    /// The values `vorticity_rgba` colours: vorticity divided by the
    /// saturation scale, unclamped, one per block in the image's order (top
    /// row = highest y).
    pub fn vorticity_values(&self, vmax: f32) -> Vec<f32> {
        let (o, vmax) = self.scaled_vorticity(vmax);
        self.image_order(|b| o[b] / vmax)
    }

    /// One byte per block in the image's order: 1 where `vorticity_rgba`
    /// paints an obstacle.
    pub fn obstacle_mask(&self) -> Vec<u8> {
        self.image_order(|b| self.solid[b] as u8)
    }

    fn scaled_vorticity(&self, vmax: f32) -> (Vec<f32>, f32) {
        let o = self.vorticity();
        let vmax = if vmax > 0.0 { vmax } else { percentile_abs(&o, 0.99).max(1e-6) };
        (o, vmax)
    }

    fn image_order<T>(&self, f: impl Fn(usize) -> T) -> Vec<T> {
        (0..self.ny).rev().flat_map(|j| (0..self.nx).map(move |i| j * self.nx + i)).map(f).collect()
    }

    /// RGBA image of speed |u| on a dark-to-light scale saturating at `vmax`.
    pub fn speed_rgba(&self, vmax: f32) -> Vec<u8> {
        let s: Vec<f32> = self.ux.iter().zip(&self.uy).map(|(a, b)| (a * a + b * b).sqrt()).collect();
        let vmax = if vmax > 0.0 { vmax } else { percentile_abs(&s, 0.99).max(1e-6) };
        self.rgba(&s, vmax, |v| sequential(v.max(0.0)))
    }

    fn rgba(&self, v: &[f32], vmax: f32, map: impl Fn(f32) -> [u8; 3]) -> Vec<u8> {
        let mut out = vec![0u8; self.nx * self.ny * 4];
        for j in 0..self.ny {
            let row = self.ny - 1 - j;
            for i in 0..self.nx {
                let b = j * self.nx + i;
                let c = if self.solid[b] { [0, 0, 0] } else { map((v[b] / vmax).clamp(-1.0, 1.0)) };
                let p = (row * self.nx + i) * 4;
                out[p..p + 3].copy_from_slice(&c);
                out[p + 3] = 255;
            }
        }
        out
    }
}

fn percentile_abs(v: &[f32], q: f32) -> f32 {
    let mut a: Vec<f32> = v.iter().map(|x| x.abs()).filter(|x| x.is_finite()).collect();
    if a.is_empty() {
        return 0.0;
    }
    let k = ((a.len() - 1) as f32 * q) as usize;
    a.select_nth_unstable_by(k, |x, y| x.partial_cmp(y).unwrap());
    a[k]
}

fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [u8; 3] {
    std::array::from_fn(|i| (a[i] + (b[i] - a[i]) * t).round().clamp(0.0, 255.0) as u8)
}

/// -1 blue, 0 white, +1 red.
fn diverging(v: f32) -> [u8; 3] {
    if v < 0.0 {
        lerp([255.0, 255.0, 255.0], [33.0, 102.0, 172.0], -v)
    } else {
        lerp([255.0, 255.0, 255.0], [178.0, 24.0, 43.0], v)
    }
}

fn sequential(v: f32) -> [u8; 3] {
    lerp([20.0, 20.0, 40.0], [250.0, 240.0, 200.0], v)
}
