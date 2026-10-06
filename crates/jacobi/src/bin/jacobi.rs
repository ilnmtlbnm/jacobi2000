//! jacobi CLI: `bench` (point-updates/s per panel, CPU and GPU), `table`
//! (max error vs the analytic solution per panel at a few iteration counts),
//! `curves` (max error over iterations for every panel on the GPU, CSV),
//! `schedules` (ternary scale schedules), `explore` (reference engines of
//! stochastic rounding and the lattice gas) and `discrete` (the converged
//! discrete solution's error against the analytic one, f64).

use std::time::Instant;

use jacobi::bitsliced::Bitsliced;
use jacobi::reference::F32Engine;
use jacobi::Problem;

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|v| v.parse().expect("number")).unwrap_or(default)
}

fn arg_str<'a>(args: &'a [String], name: &str, default: &'a str) -> &'a str {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str()).unwrap_or(default)
}

/// `--n` and `--problem quadratic|sinh`.
fn problem(args: &[String], default_n: usize) -> Problem {
    let n = jacobi::grid_size(arg(args, "--n", default_n) as f64).unwrap();
    Problem::with_case(n, jacobi::Case::parse(arg_str(args, "--problem", "quadratic")).unwrap()).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("bench") => bench(&args[1..]),
        Some("table") => table(&args[1..]),
        Some("schedules") => schedules(&args[1..]),
        Some("explore") => explore(&args[1..]),
        Some("discrete") => discrete(&args[1..]),
        Some("local") => local(&args[1..]),
        #[cfg(feature = "gpu")]
        Some("curves") => curves(&args[1..]),
        #[cfg(feature = "gpu")]
        Some("convergence") => convergence(&args[1..]),
        _ => {
            eprintln!("usage: jacobi bench [--n 512] [--iters 200] [--round 0|1] | table [--n 64] | curves [--n 128] [--iters 1048576] [--problem quadratic|sinh] [--per-octave 4] [--only f32,ternary,...] | discrete [--n 128] [--problem sinh] | convergence [--n 128] [--problem sinh] [--iters 1048576] [--only ...] [--out prefix] | local [--n 128] [--problem sinh] [--iters 1048576] [--variants global,block8,block32,geometric] | schedules [--n 128] [--iters 50000] [--short 1] [--variant 0..3] | explore [--n 128] [--iters 50000]");
            std::process::exit(2);
        }
    }
}

fn rate(n: usize, iters: usize, secs: f64) -> f64 {
    (n * n * iters) as f64 / secs
}

fn bench(args: &[String]) {
    let n = jacobi::grid_size(arg(args, "--n", 512) as f64).unwrap();
    let iters = arg(args, "--iters", 200);
    let round = arg(args, "--round", 0) == 1;
    let p = Problem::new(n).unwrap();
    println!("n={n} iters={iters} round={round}");
    let mut f = F32Engine::new(n, p.initial_f32());
    f.run(2);
    let t = Instant::now();
    f.run(iters);
    println!("cpu f32        {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    let mut f = F32Engine::new(n, p.initial_f32());
    f.red_black = true;
    f.run(2);
    let t = Instant::now();
    f.run(iters);
    println!("cpu f32 rb     {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    let mut e = jacobi::int32::Bitsliced::from_problem(&p);
    e.run(2);
    let t = Instant::now();
    e.run(iters);
    println!("cpu int32 rb   {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    for bits in jacobi::BITS {
        let mut e = Bitsliced::from_problem(&p, bits);
        e.round = round;
        e.run(2);
        let t = Instant::now();
        e.run(iters);
        println!("cpu bits {bits:2}    {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    }
    let mut e = jacobi::ternary::Bitsliced::from_problem(&p);
    e.run(2);
    let t = Instant::now();
    e.run(iters);
    println!("cpu ternary    {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    let mut e = jacobi::gas::Bitsliced::from_problem(&p, 1, true);
    e.run(2);
    let t = Instant::now();
    e.run(iters);
    println!("cpu gas        {:.3e} site-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    for bits in [4usize, 8, 16] {
        let mut e = Bitsliced::from_problem(&p, bits);
        e.stochastic = true;
        e.run(2);
        let t = Instant::now();
        e.run(iters);
        println!("cpu sr {bits:2}      {:.3e} point-updates/s", rate(n, iters, t.elapsed().as_secs_f64()));
    }
    #[cfg(feature = "gpu")]
    {
        use jacobi::gpu::GpuCtx;
        let ctx = pollster::block_on(GpuCtx::new()).unwrap();
        println!("gpu adapter: {}", ctx.adapter);
        let gpu_iters = iters * 20;
        let mut panels = gpu_panels(&ctx, &p);
        for (name, g) in panels.iter_mut() {
            g.round = round;
            g.run(&ctx, 20);
            ctx.finish();
            let t = Instant::now();
            g.run(&ctx, gpu_iters);
            ctx.finish();
            println!("gpu {name:10} {:.3e} point-updates/s ({gpu_iters} iterations)", rate(n, gpu_iters, t.elapsed().as_secs_f64()));
        }
    }
}

fn table(args: &[String]) {
    let n = jacobi::grid_size(arg(args, "--n", 64) as f64).unwrap();
    let p = Problem::new(n).unwrap();
    let marks = [10usize, 50, 200, 1000, 4000];
    println!("n={n}: max |v - analytic| over the interior");
    let mut f = F32Engine::new(n, p.initial_f32());
    let mut row = String::new();
    for &m in &marks {
        f.run(m - f.iter as usize);
        row += &format!(" it={m}:{:.3e}", p.max_error_f32(&f.cur));
    }
    println!("f32            {row}");
    for round in [false, true] {
        for bits in jacobi::BITS {
            let mut e = Bitsliced::from_problem(&p, bits);
            e.round = round;
            let mut row = String::new();
            for &m in &marks {
                e.run(m - e.iter as usize);
                row += &format!(" it={m}:{:.3e}", p.max_error_q(&e.read_q(), bits));
            }
            println!("bits {bits:2} {} {row}", if round { "round" } else { "trunc" });
        }
    }
}

/// Max error over sweeps for ternary scale schedules (plain integer
/// engine). `--short 1`: three schedules; `--variant v`: bit 0 strict dead
/// zone, bit 1 red-black order (default: all four).
fn schedules(args: &[String]) {
    use jacobi::ternary::{IntEngine, Schedule, S0};
    let n = jacobi::grid_size(arg(args, "--n", 128) as f64).unwrap();
    let iters = arg(args, "--iters", 50000);
    let p = Problem::new(n).unwrap();
    let mut marks = vec![];
    let mut m = 100;
    while m <= iters {
        marks.push(m);
        m *= 2;
    }
    let s0 = S0;
    let nn = n as u32;
    let mut list: Vec<(String, Schedule)> = vec![("adaptive".into(), Schedule::Adaptive), ("stall/2".into(), Schedule::Stall { s0, factor: 2 }), ("stall/3".into(), Schedule::Stall { s0, factor: 3 })];
    for mm in [nn, nn * nn / 32, nn * nn / 16, nn * nn / 8, nn * nn / 4] {
        list.push((format!("x1/2 M={mm}"), Schedule::Geometric { s0, factor: 2, m: mm, cycle: 0 }));
        list.push((format!("x1/3 M={mm}"), Schedule::Geometric { s0, factor: 3, m: mm, cycle: 0 }));
    }
    if arg(args, "--short", 0) == 1 {
        list = vec![("adaptive".into(), Schedule::Adaptive), ("stall/3".into(), Schedule::Stall { s0, factor: 3 }), (format!("x1/2 M={}", nn * nn / 16), Schedule::Geometric { s0, factor: 2, m: nn * nn / 16, cycle: 0 })];
    }
    let variants: Vec<(bool, bool)> = match arg(args, "--variant", 9) {
        9 => vec![(false, false), (true, false), (false, true), (true, true)],
        v => vec![(v & 1 == 1, v & 2 == 2)],
    };
    let mut head = format!("{:22}", "schedule");
    for m in &marks {
        head += &format!(" {m:>9}");
    }
    println!("n={n}\n{head}");
    let mut f = F32Engine::new(n, p.initial_f32());
    let mut row = format!("{:22}", "f32");
    for &m in &marks {
        f.run(m - f.iter as usize);
        row += &format!(" {:9.2e}", p.max_error_f32(&f.cur));
    }
    println!("{row}");
    for (strict, rb) in variants {
        println!("strict={strict} red_black={rb}");
        for (name, sch) in list.clone() {
            let mut e = IntEngine::new(&p, sch);
            e.strict = strict;
            e.red_black = rb;
            let mut row = format!("{name:22}");
            for &m in &marks {
                e.run(m - e.iter as usize);
                row += &format!(" {:9.2e}", e.max_error(&p));
            }
            println!("{row}  (last s {})", e.last_scale);
        }
    }
}

/// Reference engines: stochastic rounding and the lattice gas.
fn explore(args: &[String]) {
    use jacobi::gas::{block_error, Estimator, RefEngine};
    let n = jacobi::grid_size(arg(args, "--n", 128) as f64).unwrap();
    let iters = arg(args, "--iters", 50000);
    let p = Problem::new(n).unwrap();
    let mut marks = vec![];
    let mut m = 100;
    while m <= iters {
        marks.push(m);
        m *= 2;
    }
    println!("n={n}");
    for bits in [4usize, 8, 16] {
        let mut q = p.initial_q(bits);
        let mut t = 0;
        let mut row = format!("sr {bits:2}         ");
        for &m in &marks {
            jacobi::stochastic::int_run(n, &mut q, m - t, 1, t as u32);
            t = m;
            row += &format!(" {:9.2e}", p.max_error_q(&q, bits));
        }
        println!("{row}");
    }
    for ann in [false, true] {
        let mut e = RefEngine::new(&p, 1, ann);
        let mut est = Estimator::new(n);
        let mut row = format!("gas ann={ann:5}  ");
        let mut k = 0;
        while (e.iter as usize) < iters {
            let step = (e.iter as usize / 8).max(16);
            e.run(step);
            let r = est.update(e.iter, &e.acc);
            if k < marks.len() && e.iter as usize >= marks[k] {
                let (v, _) = r.unwrap();
                row += &format!(" {:9.2e}", block_error(&p, &v));
                k += 1;
            }
        }
        let occ = e.cur.iter().map(|s| s.count_ones()).sum::<u32>() as f64 / (4 * n * n) as f64;
        println!("{row}  occupancy {occ:.3}");
    }
}

/// The page's panels on the GPU, in page order.
#[cfg(feature = "gpu")]
fn gpu_panels(ctx: &jacobi::gpu::GpuCtx, p: &Problem) -> Vec<(String, jacobi::gpu::GpuPanel)> {
    use jacobi::gpu::GpuPanel;
    let n = p.n;
    let mut v = vec![("f32".to_string(), GpuPanel::f32(ctx, p).unwrap())];
    v.push(("f32 rb".into(), GpuPanel::f32_red_black(ctx, p).unwrap()));
    v.push(("int32 rb".into(), GpuPanel::int32_red_black(ctx, n, &jacobi::ternary::initial(p)).unwrap()));
    v.push(("ternary".into(), GpuPanel::ternary(ctx, n, &jacobi::ternary::initial(p)).unwrap()));
    v.push(("lattice gas".into(), GpuPanel::gas(ctx, p, &jacobi::gas::initial(n), 1, true).unwrap()));
    for bits in [4usize, 8, 16] {
        v.push((format!("sr {bits}"), GpuPanel::stochastic(ctx, n, bits, &p.initial_q(bits), 1).unwrap()));
    }
    for bits in jacobi::BITS {
        v.push((format!("bits {bits}"), GpuPanel::bits(ctx, p, bits).unwrap()));
    }
    v
}

/// Max error over iterations for every panel on the GPU (or the `--only`
/// ones, comma-separated names), at iteration counts 2^(k/per_octave); CSV
/// `n,panel,iteration,max_error,gpu_seconds` on stdout, with a `problem`
/// column first when `--problem` is given (seconds: the panel's iterations
/// alone, to GPU completion, without the readbacks). `--discrete 1` adds a
/// `max_error_discrete` column, the error against the discrete solution
/// (`reference::discrete_solution`); `--stop-flat 1` stops a panel once its
/// iteration count is past twice the one where its lowest error so far
/// (against the discrete solution when given) was first reached, and past
/// n^2 / 4 (the early plateaus of the ternary panel end before that).
#[cfg(feature = "gpu")]
fn curves(args: &[String]) {
    use jacobi::gpu::{GpuCtx, Kind};
    let iters = arg(args, "--iters", 1 << 20) as u64;
    let p = problem(args, 128);
    let n = p.n;
    let per_octave = arg(args, "--per-octave", 4) as f64;
    let only: Vec<&str> = arg_str(args, "--only", "").split(',').filter(|s| !s.is_empty()).collect();
    let tag = if args.iter().any(|a| a == "--problem") { format!("{},", p.case.name()) } else { String::new() };
    let disc = (arg(args, "--discrete", 0) == 1).then(|| jacobi::reference::discrete_solution(&p));
    let stop_flat = arg(args, "--stop-flat", 0) == 1;
    let ctx = pollster::block_on(GpuCtx::new()).unwrap();
    eprintln!("gpu adapter: {}", ctx.adapter);
    let mut marks: Vec<u64> = (0..).map(|k| (2f64.powf(k as f64 / per_octave)).round() as u64).take_while(|&m| m <= iters).collect();
    marks.dedup();
    println!("{}n,panel,iteration,max_error,{}gpu_seconds", if tag.is_empty() { "" } else { "problem," }, if disc.is_some() { "max_error_discrete," } else { "" });
    for (name, mut g) in gpu_panels(&ctx, &p) {
        if !only.is_empty() && !only.contains(&name.as_str()) {
            continue;
        }
        let mut est = jacobi::gas::Estimator::new(n);
        let mut secs = 0.0;
        let (mut best, mut best_at) = (f64::INFINITY, 0u64);
        for &m in &marks {
            let t = Instant::now();
            g.run(&ctx, (m - g.iter) as usize);
            ctx.finish();
            secs += t.elapsed().as_secs_f64();
            let field: Vec<f64> = match g.kind {
                Kind::F32 | Kind::F32RedBlack => pollster::block_on(g.read_f32(&ctx)).into_iter().map(|x| x as f64).collect(),
                Kind::Bits(b) | Kind::Stochastic(b) => pollster::block_on(g.read_q(&ctx)).into_iter().map(|x| x as f64 / jacobi::qmax(b) as f64).collect(),
                Kind::Ternary | Kind::Int32RedBlack => pollster::block_on(g.read_planes(&ctx, jacobi::ternary::W)).into_iter().map(|x| x as i32 as f64 / jacobi::ternary::Q as f64).collect(),
                Kind::Gas => vec![],
            };
            let err = match g.kind {
                Kind::Gas => {
                    let acc: Vec<i32> = pollster::block_on(g.read_aux(&ctx)).into_iter().map(|x| x as i32).collect();
                    match est.update(g.iter, &acc) {
                        Some((b, _)) => jacobi::gas::block_error(&p, &b),
                        None => continue,
                    }
                }
                _ => p.max_error(|k| field[k]),
            };
            let mut col = String::new();
            let mut tracked = err;
            if let (Some(d), false) = (&disc, field.is_empty()) {
                let e = jacobi::reference::max_diff_interior(n, &field, d);
                col = format!("{e:.6e},");
                tracked = e;
            }
            println!("{tag}{n},{name},{m},{err:.6e},{col}{secs:.4}");
            if tracked < best {
                (best, best_at) = (tracked, m);
            }
            if stop_flat && m > (n * n / 4) as u64 && m > 2 * best_at {
                break;
            }
        }
        eprintln!("{name}: {} iterations, {secs:.2} s", g.iter);
    }
}

/// The discrete solution's own error against the analytic one: the floor no
/// solver of the 5-point stencil can go below.
fn discrete(args: &[String]) {
    let p = problem(args, 128);
    let u = jacobi::reference::discrete_solution(&p);
    println!("n={} problem={} max |discrete - analytic| = {:.4e}", p.n, p.case.name(), p.max_error(|k| u[k]));
}

/// Ternary red-black on the CPU integer engine with the scale from the
/// global residual OR ("global", as on the page), from each b x b block's
/// own OR ("block<b>", no global reduction), or a fixed schedule
/// ("geometric": x1/2 every n^2 / 16 sweeps from S0, no reduction at all).
/// CSV `problem,n,panel,iteration,max_error,max_error_discrete,cpu_seconds`
/// at 2^(k/16) sweeps, with the `curves --stop-flat 1` rule.
fn local(args: &[String]) {
    use jacobi::ternary::{initial, sweep_blocks, IntEngine, Schedule, Q, S0};
    let p = problem(args, 128);
    let n = p.n;
    let iters = arg(args, "--iters", 1 << 20) as u64;
    let disc = jacobi::reference::discrete_solution(&p);
    let mut marks: Vec<u64> = (0..).map(|k| (2f64.powf(k as f64 / 16.0)).round() as u64).take_while(|&m| m <= iters).collect();
    marks.dedup();
    println!("problem,n,panel,iteration,max_error,max_error_discrete,cpu_seconds");
    for v in arg_str(args, "--variants", "global,block8,block32,geometric").split(',') {
        let mut e = IntEngine::new(&p, if v == "geometric" { Schedule::Geometric { s0: S0, factor: 2, m: (n * n / 16) as u32, cycle: 0 } } else { Schedule::Adaptive });
        e.red_black = true;
        let b: usize = v.strip_prefix("block").map(|x| x.parse().unwrap()).unwrap_or(0);
        let mut ors = vec![0u32; n.div_ceil(b.max(1)).pow(2)];
        let mut other = e.cur.clone();
        let mut u = initial(&p);
        let (mut best, mut best_at, mut secs) = (f64::INFINITY, 0u64, 0.0);
        let mut it = 0u64;
        for &m in &marks {
            let t = Instant::now();
            if b > 0 {
                while it < m {
                    sweep_blocks(n, &mut u, &mut other, b, &mut ors, it == 0);
                    it += 1;
                }
            } else {
                e.run((m - e.iter) as usize);
                u.clone_from(&e.cur);
            }
            secs += t.elapsed().as_secs_f64();
            let f: Vec<f64> = u.iter().map(|&x| x as f64 / Q as f64).collect();
            let err = p.max_error(|k| f[k]);
            let ed = jacobi::reference::max_diff_interior(n, &f, &disc);
            println!("{},{n},ternary {v},{m},{err:.6e},{ed:.6e},{secs:.4}", p.case.name());
            if ed < best {
                (best, best_at) = (ed, m);
            }
            if m > (n * n / 4) as u64 && m > 2 * best_at {
                break;
            }
        }
        eprintln!("{v}: n={n} best {best:.3e} at {best_at}, {secs:.1} s");
    }
}

/// Field of a GPU panel as v per point (not for the lattice gas).
#[cfg(feature = "gpu")]
fn gpu_field(ctx: &jacobi::gpu::GpuCtx, g: &jacobi::gpu::GpuPanel) -> Vec<f64> {
    use jacobi::gpu::Kind;
    match g.kind {
        Kind::F32 | Kind::F32RedBlack => pollster::block_on(g.read_f32(ctx)).into_iter().map(|x| x as f64).collect(),
        Kind::Bits(b) | Kind::Stochastic(b) => pollster::block_on(g.read_q(ctx)).into_iter().map(|x| x as f64 / jacobi::qmax(b) as f64).collect(),
        Kind::Ternary | Kind::Int32RedBlack => pollster::block_on(g.read_planes(ctx, jacobi::ternary::W)).into_iter().map(|x| x as i32 as f64 / jacobi::ternary::Q as f64).collect(),
        Kind::Gas => panic!("no field for the lattice gas"),
    }
}

/// Error spectra in the Dirichlet sine modes sin(pi p i / N) sin(pi q j / N),
/// N = n - 1, p, q = 1..N-1, of the error against the discrete solution.
struct Spectra {
    m: usize,
    big_n: f64,
    /// sines[p - 1][i - 1].
    sines: Vec<f64>,
}

impl Spectra {
    fn new(n: usize) -> Self {
        let m = n - 2;
        let big_n = (n - 1) as f64;
        let sines = (0..m * m).map(|k| (std::f64::consts::PI * ((k / m) + 1) as f64 * ((k % m) + 1) as f64 / big_n).sin()).collect();
        Spectra { m, big_n, sines }
    }

    /// Interior error (row-major m x m).
    fn error(&self, f: &[f64], disc: &[f64]) -> Vec<f64> {
        let (m, n) = (self.m, self.m + 2);
        (0..m * m).map(|k| f[(k / m + 1) * n + k % m + 1] - disc[(k / m + 1) * n + k % m + 1]).collect()
    }

    /// Coefficients c[p][q] (row-major), e = sum c sin sin.
    fn coefficients(&self, e: &[f64]) -> Vec<f64> {
        let m = self.m;
        let s = &self.sines;
        // t[p][j] = sum_i s[p][i] e[i][j]
        let mut t = vec![0f64; m * m];
        for p in 0..m {
            let row = &mut t[p * m..(p + 1) * m];
            for i in 0..m {
                let a = s[p * m + i];
                let er = &e[i * m..(i + 1) * m];
                for (r, x) in row.iter_mut().zip(er) {
                    *r += a * x;
                }
            }
        }
        let scale = (2.0 / self.big_n).powi(2);
        let mut c = vec![0f64; m * m];
        for p in 0..m {
            for q in 0..m {
                let tr = &t[p * m..(p + 1) * m];
                let sq = &s[q * m..(q + 1) * m];
                c[p * m + q] = scale * tr.iter().zip(sq).map(|(a, b)| a * b).sum::<f64>();
            }
        }
        c
    }

    /// One coefficient.
    fn mode(&self, e: &[f64], p: usize, q: usize) -> f64 {
        let m = self.m;
        let mut acc = 0.0;
        for i in 0..m {
            let si = self.sines[(p - 1) * m + i];
            let er = &e[i * m..(i + 1) * m];
            acc += si * er.iter().zip(&self.sines[(q - 1) * m..q * m]).map(|(a, b)| a * b).sum::<f64>();
        }
        acc * (2.0 / self.big_n).powi(2)
    }

    /// Jacobi factor per sweep of mode (p, q).
    fn mu(&self, p: usize, q: usize) -> f64 {
        use std::f64::consts::PI;
        ((PI * p as f64 / self.big_n).cos() + (PI * q as f64 / self.big_n).cos()) / 2.0
    }

    /// Log-spaced radial bins of |k| = sqrt(p^2 + q^2), 8 per octave: bin index per mode.
    fn bin(p: usize, q: usize) -> usize {
        ((((p * p + q * q) as f64).sqrt().log2() * 8.0).floor()) as usize
    }
}

/// Convergence measurements of the panels (GPU; "ternary block8" on the
/// CPU integer engine): at 2^(k/16) sweeps (and sweep 0), max and L2 error
/// against the analytic and the discrete solution, L2 residual
/// |E + W + N + S - 4 C| / 4 (in v), the sine-mode amplitudes (1, 1) and
/// (n/8 + 1, n/8 + 1) of the error against the discrete solution, and at sweeps 0
/// and 4^k the radially binned error spectrum with the textbook Jacobi and
/// red-black Gauss-Seidel predictions from the sweep-0 spectrum. Writes
/// `<out>-metrics.csv` and `<out>-spectra.csv`; stops a panel by the
/// `curves --stop-flat 1` rule on the max error against the discrete solution.
#[cfg(feature = "gpu")]
fn convergence(args: &[String]) {
    use jacobi::gpu::{GpuCtx, GpuPanel};
    use std::io::Write;
    let p = problem(args, 128);
    let n = p.n;
    let iters = arg(args, "--iters", 1 << 20) as u64;
    let out = arg_str(args, "--out", "conv");
    let only: Vec<String> = arg_str(args, "--only", "f32,f32 rb,int32 rb,ternary,sr 16,ternary block8").split(',').map(|s| s.to_string()).collect();
    let disc = jacobi::reference::discrete_solution(&p);
    let sp = Spectra::new(n);
    let m = sp.m;
    // Odd, so it is present in the even-symmetric start of both problems.
    let mid = n / 8 + 1;
    let mut marks: Vec<u64> = (0..).map(|k| (2f64.powf(k as f64 / 16.0)).round() as u64).take_while(|&x| x <= iters).collect();
    marks.dedup();
    marks.insert(0, 0);
    let spectral = |it: u64| it == 0 || (it.is_power_of_two() && it.trailing_zeros().is_multiple_of(2));
    let mut fm = std::fs::File::create(format!("{out}-metrics.csv")).unwrap();
    let mut fs = std::fs::File::create(format!("{out}-spectra.csv")).unwrap();
    writeln!(fm, "problem,n,panel,iteration,seconds,max_error,l2_error,max_error_discrete,l2_error_discrete,l2_residual,amp_1_1,amp_mid,mid").unwrap();
    writeln!(fs, "problem,n,panel,iteration,k,modes,energy_per_mode,jacobi_theory,gauss_seidel_theory").unwrap();
    let ctx = only.iter().any(|s| s != "ternary block8").then(|| pollster::block_on(GpuCtx::new()).unwrap());
    let nbins = Spectra::bin(m, m) + 1;
    let mut count = vec![0usize; nbins];
    for pp in 1..=m {
        for q in 1..=m {
            count[Spectra::bin(pp, q)] += 1;
        }
    }
    for name in &only {
        let mut g: Option<GpuPanel> = ctx.as_ref().and_then(|ctx| gpu_panels(ctx, &p).into_iter().find(|(nm, _)| nm == name).map(|(_, g)| g));
        assert!(g.is_some() || name == "ternary block8", "unknown panel {name}");
        let (mut u, mut other) = (jacobi::ternary::initial(&p), jacobi::ternary::initial(&p));
        let mut ors = vec![0u32; (n / 8).pow(2)];
        let mut it = 0u64;
        let (mut secs, mut best, mut best_at) = (0.0, f64::INFINITY, 0u64);
        let mut e0: Option<Vec<f64>> = None;
        for &mk in &marks {
            let t = Instant::now();
            match g.as_mut() {
                Some(g) => {
                    let ctx = ctx.as_ref().unwrap();
                    g.run(ctx, (mk - g.iter) as usize);
                    ctx.finish();
                }
                None => {
                    while it < mk {
                        jacobi::ternary::sweep_blocks(n, &mut u, &mut other, 8, &mut ors, it == 0);
                        it += 1;
                    }
                }
            }
            secs += t.elapsed().as_secs_f64();
            let f: Vec<f64> = match g.as_ref() {
                Some(g) => gpu_field(ctx.as_ref().unwrap(), g),
                None => u.iter().map(|&x| x as f64 / jacobi::ternary::Q as f64).collect(),
            };
            let mut res = 0f64;
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    let k = i * n + j;
                    let r = (f[k + 1] + f[k - 1] + f[k - n] + f[k + n] - 4.0 * f[k]) / 4.0;
                    res += r * r;
                }
            }
            let l2_res = (res / (m * m) as f64).sqrt();
            let e = sp.error(&f, &disc);
            let max_d = e.iter().fold(0f64, |a, x| a.max(x.abs()));
            let l2_d = (e.iter().map(|x| x * x).sum::<f64>() / (m * m) as f64).sqrt();
            let mut sa = 0f64;
            for i in 1..n - 1 {
                for j in 1..n - 1 {
                    sa += (f[i * n + j] - p.analytic(i, j)).powi(2);
                }
            }
            let l2_a = (sa / (m * m) as f64).sqrt();
            let max_a = p.max_error(|k| f[k]);
            writeln!(fm, "{},{n},{name},{mk},{secs:.4},{max_a:.6e},{l2_a:.6e},{max_d:.6e},{l2_d:.6e},{l2_res:.6e},{:.6e},{:.6e},{mid}", p.case.name(), sp.mode(&e, 1, 1), sp.mode(&e, mid, mid)).unwrap();
            if spectral(mk) {
                let c = sp.coefficients(&e);
                let e0c = e0.get_or_insert_with(|| c.clone());
                let mut energy = vec![0f64; nbins];
                let mut jac = vec![0f64; nbins];
                let mut gs = vec![0f64; nbins];
                for pp in 1..=m {
                    for q in 1..=m {
                        let b = Spectra::bin(pp, q);
                        let k = (pp - 1) * m + q - 1;
                        let mu2 = sp.mu(pp, q).powi(2);
                        energy[b] += c[k] * c[k];
                        let e00 = e0c[k] * e0c[k];
                        jac[b] += e00 * mu2.powf(mk as f64);
                        gs[b] += e00 * mu2.powf(2.0 * mk as f64);
                    }
                }
                for b in 0..nbins {
                    if count[b] > 0 {
                        let kc = 2f64.powf((b as f64 + 0.5) / 8.0);
                        let cn = count[b] as f64;
                        writeln!(fs, "{},{n},{name},{mk},{kc:.4},{},{:.6e},{:.6e},{:.6e}", p.case.name(), count[b], energy[b] / cn, jac[b] / cn, gs[b] / cn).unwrap();
                    }
                }
            }
            if max_d < best {
                (best, best_at) = (max_d, mk);
            }
            if mk > (n * n / 4) as u64 && mk > 2 * best_at {
                break;
            }
        }
        eprintln!("{name}: n={n} {} best {best:.3e} at {best_at}, {secs:.1} s", p.case.name());
    }
}
