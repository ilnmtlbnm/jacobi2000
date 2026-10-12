//! Roadmap item 1 (Lagrangian formulation): dump raw SPH particle
//! trajectories (position, velocity, mass, every tick) to disk, so a
//! PyTorch particle model can train on particles directly instead of the
//! splatted-to-grid fields (`training/jacobi2000_pt/lagrangian.py`).
//!
//! One file per trajectory: a JSON header (particle count, ticks, dt, g,
//! rest_density) plus a flat little-endian f32 file laid out
//! `[ticks, 5, n]` (x, y, vx, vy, mass), matching the row-major convention
//! `training/jacobi2000_pt/data.py` already reads for the grid dataset.
//! Ticks are recorded every `SPH_STEPS_PER_TICK` SPH steps (the same tick
//! definition as the splatted-grid data), so a Lagrangian model
//! ticks at the same physical rate as the grid model would.

use std::fs;
use std::io::Write;

use solvers::sph::{Initial, Params, Sph};

const STEPS_PER_TICK: usize = 9;

struct Trajectory {
    name: String,
    initial: Initial,
    ticks: usize,
}

fn dump_one(traj: Trajectory, params: Params, out_dir: &str) {
    let mut sph = Sph::new(params, traj.initial);
    let n = sph.len();
    let mut buf: Vec<f32> = Vec::with_capacity(traj.ticks * 5 * n);

    let record = |sph: &Sph, buf: &mut Vec<f32>| {
        let (x, y) = sph.positions();
        let (vx, vy) = sph.velocities();
        let m = sph.masses();
        buf.extend_from_slice(x);
        buf.extend_from_slice(y);
        buf.extend_from_slice(vx);
        buf.extend_from_slice(vy);
        buf.extend_from_slice(m);
    };

    record(&sph, &mut buf);
    for _ in 0..traj.ticks - 1 {
        sph.steps(STEPS_PER_TICK);
        record(&sph, &mut buf);
    }

    let mut bytes = Vec::with_capacity(buf.len() * 4);
    for v in &buf {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(format!("{out_dir}/traj-{}.f32", traj.name), &bytes).unwrap();

    let header = format!(
        "{{\"name\":\"{}\",\"n\":{n},\"ticks\":{},\"dt_effective\":{},\"g\":{},\"rest_density\":{},\"nx\":{},\"ny\":{},\"dx\":{}}}",
        traj.name,
        traj.ticks,
        STEPS_PER_TICK as f32 * params.dt,
        params.gravity,
        params.rest_density,
        params.nx,
        params.ny,
        params.dx,
    );
    fs::write(format!("{out_dir}/traj-{}.json", traj.name), header).unwrap();
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "data/sph".to_string());
    fs::create_dir_all(&out_dir).unwrap();

    let params = Params::default();
    let mut trajectories = vec![
        Trajectory { name: "dambreak".to_string(), initial: Initial::DamBreak, ticks: 60 },
        Trajectory { name: "drop".to_string(), initial: Initial::Drop, ticks: 60 },
        Trajectory {
            name: "dambreak-small".to_string(),
            initial: Initial::DamBreakSized { col_w_frac: 0.15, col_h_frac: 0.3 },
            ticks: 60,
        },
        Trajectory {
            name: "drop-small".to_string(),
            initial: Initial::DropSized { cx_frac: 0.35, radius_frac: 0.06, layer_frac: 0.15 },
            ticks: 60,
        },
    ];
    // A handful more, varied column/blob sizes and drop positions, so the
    // real (not smoke) training run has more than 4 trajectories to sample
    // from.
    for (i, (w, h)) in [(0.2, 0.4), (0.3, 0.35), (0.18, 0.25)].iter().enumerate() {
        trajectories.push(Trajectory {
            name: format!("dambreak-v{i}"),
            initial: Initial::DamBreakSized { col_w_frac: *w, col_h_frac: *h },
            ticks: 60,
        });
    }
    for (i, (cx, r, layer)) in [(0.3, 0.08, 0.18), (0.6, 0.1, 0.22), (0.5, 0.05, 0.12)].iter().enumerate() {
        trajectories.push(Trajectory {
            name: format!("drop-v{i}"),
            initial: Initial::DropSized { cx_frac: *cx, radius_frac: *r, layer_frac: *layer },
            ticks: 60,
        });
    }

    let mut manifest = fs::File::create(format!("{out_dir}/manifest.txt")).unwrap();
    for traj in trajectories {
        let name = traj.name.clone();
        dump_one(traj, params, &out_dir);
        writeln!(manifest, "{}", name).unwrap();
        println!("wrote {}", name);
    }
}
