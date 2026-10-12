//! Times `GpuModel::step` (upload, one submit, readback, host add, mass
//! projection: the work `crates/web` does per tick, minus the reference
//! solver). Usage: `bench <checkpoint-dir> <steps> <repeats>`.

use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&a[1]);
    let steps: usize = a[2].parse().unwrap();
    let reps: usize = a[3].parse().unwrap();
    let model = engine::Model::load(&dir);
    let cfg = model.cfg.clone();
    let gpu = pollster::block_on(model.to_gpu()).expect("no adapter");
    let (c, ny, nx) = (cfg.channels, cfg.ny, cfg.nx);
    let mut x0 = vec![0.0f32; c * ny * nx];
    for ch in 0..c {
        for y in 0..ny {
            for x in 0..nx {
                x0[(ch * ny + y) * nx + x] = ((ch + 1) as f32 * 0.05)
                    * (std::f32::consts::TAU * (ch + 1) as f32 * x as f32 / nx as f32).sin()
                    * (std::f32::consts::TAU * y as f32 / ny as f32).cos();
            }
        }
    }
    let meta = engine::meta_of(&cfg.params()).to_vec();
    let n = ny * nx;
    let mass0 = engine::project::mass(&x0, n);
    let mut times = vec![];
    let mut checksum = 0.0f32;
    for r in 0..reps + 1 {
        let mut x = x0.clone();
        let t = Instant::now();
        for _ in 0..steps {
            let mut next = pollster::block_on(gpu.step(&x, &meta));
            engine::project::project_mass(&mut next, mass0, n);
            x = next;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
        checksum = x.iter().sum();
        if r > 0 {
            times.push(ms);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "engine-webgpu step: median {:.3} min {:.3} max {:.3} ms/step (steps {steps}, repeats {reps}, checksum {checksum:.6})",
        times[times.len() / 2], times[0], times[times.len() - 1]
    );
}
