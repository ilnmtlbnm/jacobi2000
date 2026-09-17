use solvers::{Params, ShallowWater, State};

fn main() {
    let p = Params { nu: 1e-3, ..Default::default() };
    let mut sw = ShallowWater::new(p);
    let mut s = State::flat(&p);
    s.add_drop(&p, 24.0, 40.0, 0.05, 4.0);
    let n = 20_000;
    let t = std::time::Instant::now();
    sw.steps(&mut s, n);
    let secs = t.elapsed().as_secs_f64();
    println!("{}x{}: {n} ticks in {secs:.3}s = {:.0} ticks/s", p.nx, p.ny, n as f64 / secs);
}
