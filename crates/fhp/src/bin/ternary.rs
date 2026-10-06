//! Ternary FHP measurements against FHP-III: `theory`, `bench`, `shear`,
//! `advect`, `noise`, `street`. Results are printed as CSV.

#![allow(clippy::needless_range_loop)]

use fhp::ternary::measure::{self, Model};
use fhp::ternary::{self, Rule};
use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn list<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Vec<T> {
    arg(args, name, default.to_string())
        .split(',')
        .filter_map(|x| x.parse().ok())
        .collect()
}

fn model(s: &str) -> Model {
    match s {
        "fhp3" => Model::Fhp3,
        "ternary" => Model::Ternary(Rule::Other),
        "ternary-uniform" => Model::Ternary(Rule::Uniform),
        "ternary-optimal" => Model::Ternary(Rule::Optimal),
        _ if s.starts_with("ternary-") => {
            Model::Ternary(Rule::parse(&s[8..]).expect("ternary-other|uniform|optimal|mixK"))
        }
        _ => panic!("model: fhp3 | ternary | ternary-uniform"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = &args[1.min(args.len())..];
    match args.first().map(|s| s.as_str()) {
        Some("theory") => theory(rest),
        Some("bench") => bench(rest),
        Some("shear") => shear(rest),
        Some("advect") => advect(rest),
        Some("noise") => noise(rest),
        Some("street") => street(rest),
        Some("summarize") => summarize(rest),
        Some("gen") => gen(rest),
        _ => {
            eprintln!(
                "usage: ternary theory | bench | shear | advect | noise | street | summarize | gen"
            );
            std::process::exit(2);
        }
    }
}

/// Boltzmann-approximation g, nu and noise per density, for FHP-III and the
/// two ternary rules; then g for ternary moving channels with a larger rest
/// capacity.
fn theory(args: &[String]) {
    let ds: Vec<f64> = list(
        args,
        "--d",
        "0.05,0.1,0.15,0.2,0.25,0.28,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9",
    );
    let t = ternary::Table::new();
    let sizes = t.class_sizes();
    let mut hist = std::collections::BTreeMap::new();
    for &q in &sizes {
        *hist.entry(q).or_insert(0) += 1;
    }
    eprintln!(
        "ternary classes: {} classes, max size {}, size histogram {:?}",
        sizes.len(),
        sizes.iter().max().unwrap(),
        hist
    );
    println!("d,model,g,nu,g_over_nu,var,noise16");
    for &d in &ds {
        let mut rows = vec![];
        if d < 0.5 {
            rows.push(("fhp3", ternary::g_theory(1, 1, d), ternary::nu_fhp3(d), 1));
        }
        rows.push((
            "ternary",
            ternary::g_theory(2, 2, d),
            ternary::nu_ternary(d, Rule::Other),
            2,
        ));
        rows.push((
            "ternary-uniform",
            ternary::g_theory(2, 2, d),
            ternary::nu_ternary(d, Rule::Uniform),
            2,
        ));
        rows.push((
            "ternary-optimal",
            ternary::g_theory(2, 2, d),
            ternary::nu_ternary(d, Rule::Optimal),
            2,
        ));
        rows.push((
            "ternary-chiral",
            ternary::g_theory(2, 2, d),
            ternary::nu_ternary(d, Rule::Chiral),
            2,
        ));
        for (name, g, nu, m) in rows {
            let var = ternary::channel_moments(m, ternary::fugacity(m, m, d)).1;
            let n16 = (3.0 * var / 256.0).sqrt() / (7.0 * d);
            println!("{d},{name},{g:.5},{nu:.5},{:.4},{var:.5},{n16:.5}", g / nu);
        }
    }
    // Peak g/nu over a fine density grid.
    for (name, m) in [
        ("fhp3", 1u32),
        ("ternary", 2),
        ("ternary-uniform", 2),
        ("ternary-optimal", 2),
        ("ternary-chiral", 2),
    ] {
        let top = if m == 1 { 0.5 } else { 1.0 };
        let mut best = (0.0, 0.0, 0.0, 0.0);
        let mut d = 0.02;
        while d < top {
            let g = ternary::g_theory(m, m, d);
            let nu = match name {
                "fhp3" => ternary::nu_fhp3(d),
                "ternary" => ternary::nu_ternary(d, Rule::Other),
                "ternary-optimal" => ternary::nu_ternary(d, Rule::Optimal),
                "ternary-chiral" => ternary::nu_ternary(d, Rule::Chiral),
                _ => ternary::nu_ternary(d, Rule::Uniform),
            };
            if g / nu > best.0 {
                best = (g / nu, d, g, nu);
            }
            d += 0.01;
        }
        eprintln!(
            "{name}: max g/nu {:.3} at d {:.2} (g {:.4}, nu {:.4})",
            best.0, best.1, best.2, best.3
        );
    }
    // g with nr rest channels of capacity 2 (Boghosian et al. section VII):
    // g = ((6 + nr) / 12) f k3 / var^2 at mean count f per channel.
    for nr in [1u32, 2, 4, 6] {
        let (mut best, mut d) = ((0.0, 0.0), 0.005);
        let mut first_one = None;
        while d < 1.0 {
            let z = ternary::fugacity(2, 2, d);
            let (f, v, k3) = ternary::channel_moments(2, z);
            let g = (6 + nr) as f64 / 12.0 * f * k3 / (v * v);
            if g > best.0 {
                best = (g, d);
            }
            if g >= 1.0 {
                first_one.get_or_insert(d);
            }
            d += 0.005;
        }
        eprintln!("capacity 2, {nr} rest channels: max g {:.4} at mean count {:.3} per channel; g >= 1 from {:?}", best.0, best.1, first_one);
    }
}

fn bench(args: &[String]) {
    use fhp::bitsliced::Bitsliced;
    use fhp::ternary::engine::{self, Ternary};
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 512usize);
    let steps = arg(args, "--steps", 100usize);
    let sites = (w * h) as f64;
    let report = |name: &str, n: usize, secs: f64| println!("{name:<34} {w}x{h} {n:>6} steps {secs:>8.3} s {:>10.1} M site-updates/s", sites * n as f64 / secs / 1e6);
    // Periodic lattices at the street densities, no obstacle.
    let fcells = fhp::initial_state(w, h, 0.28, 0.2, &fhp::Obstacle::None, 1);
    let mean = ternary::equilibrium(2, 2, 0.6, 0.2, 0.0);
    let tcells: Vec<u16> = (0..w * h).map(|i| engine::sample_equilibrium(&mean, 1, i as u32)).collect();
    let fp = fhp::Params { seed: 1, ..Default::default() };
    let tp = |rule| engine::Params { seed: 1, rule, boundary: None };
    if !args.iter().any(|a| a == "--gpu-only") {
        for threads in [1, rayon::current_num_threads()] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let n = (steps / 4).max(4);
                let mut r = fhp::reference::Reference::new(w, h, fcells.clone(), fp);
                if threads == 1 {
                    let t0 = Instant::now();
                    r.run(n);
                    report("fhp3 reference (1 thread)", n, t0.elapsed().as_secs_f64());
                }
                for rule in [Rule::Other, Rule::Chiral] {
                    let mut t = Ternary::new(w, h, tcells.clone(), tp(rule));
                    t.run(1);
                    let t0 = Instant::now();
                    t.run(n);
                    report(&format!("ternary reference {rule:?} ({threads} thr)"), n, t0.elapsed().as_secs_f64());
                }
                let mut b = Bitsliced::from_cells(w, h, &fcells, fp);
                b.run(2);
                let t0 = Instant::now();
                b.run(steps);
                report(&format!("fhp3 bitsliced ({threads} thr)"), steps, t0.elapsed().as_secs_f64());
                let mut b = fhp::ternary::bitsliced::Bitsliced::from_cells(w, h, &tcells, tp(Rule::Chiral));
                b.run(2);
                let t0 = Instant::now();
                b.run(n);
                report(&format!("ternary bitsliced ({threads} thr)"), n, t0.elapsed().as_secs_f64());
            });
        }
    }
    if args.iter().any(|a| a == "--gpu" || a == "--gpu-only") {
        let gsteps = arg(args, "--gpu-steps", 2000usize);
        let mut g = pollster::block_on(fhp::gpu::Gpu::new(w, h, &fcells, fp)).expect("GPU adapter");
        g.run(50);
        g.finish();
        let t0 = Instant::now();
        g.run(gsteps);
        g.finish();
        report(&format!("fhp3 gpu {}", g.adapter), gsteps, t0.elapsed().as_secs_f64());
        let mut g = pollster::block_on(fhp::ternary::gpu::Gpu::new(w, h, &tcells, 1)).expect("GPU adapter");
        g.run(50);
        g.finish();
        let n = gsteps / 4;
        let t0 = Instant::now();
        g.run(n);
        g.finish();
        report(&format!("ternary gpu bitsliced {}", g.adapter), n, t0.elapsed().as_secs_f64());
        let mut g = pollster::block_on(fhp::ternary::gpu::Gpu::with_layout(w, h, &tcells, 1, fhp::ternary::gpu::Layout::Table)).expect("GPU adapter");
        g.run(50);
        g.finish();
        let t0 = Instant::now();
        g.run(gsteps);
        g.finish();
        report(&format!("ternary gpu table {}", g.adapter), gsteps, t0.elapsed().as_secs_f64());
    }
}

/// Shear-wave decay: for each model, density, height (one wavelength per
/// height, mode m) and seed, the fitted viscosity. With --series, also the
/// amplitude series of the first seed.
fn shear(args: &[String]) {
    let models: Vec<String> = list(args, "--model", "fhp3,ternary");
    let ds: Vec<f64> = list(args, "--d", "0.1,0.2,0.28,0.35,0.42");
    let hs: Vec<usize> = list(args, "--h", "64,128");
    let w = arg(args, "--width", 256usize);
    let m = arg(args, "--m", 1usize);
    let u0 = arg(args, "--u0", 0.05f64);
    let seeds = arg(args, "--seeds", 3u32);
    let floor = arg(args, "--floor", 0.3f64);
    let series = arg(args, "--series", String::new());
    println!("model,d,width,height,m,k,seed,steps,nu,nu_se,nu_theory");
    let mut out = String::new();
    for mn in &models {
        let md = model(mn);
        for &d in &ds {
            if md == Model::Fhp3 && d >= 0.5 {
                continue;
            }
            let nu_th = match md {
                Model::Fhp3 => ternary::nu_fhp3(d),
                Model::Ternary(r) => ternary::nu_ternary(d, r),
            };
            for &h in &hs {
                let k = 2.0 * std::f64::consts::PI * m as f64 / (h as f64 * 3f64.sqrt() / 2.0);
                // Run until the amplitude should reach floor/1.3 (or 4000 steps at least).
                let steps = (((1.0 / floor).ln() * 1.3 / (nu_th.max(0.005) * k * k)) as usize)
                    .clamp(2000, 200_000);
                let every = (steps / 200).max(1);
                for seed in 1..=seeds {
                    let t0 = Instant::now();
                    let s = measure::shear_series(md, w, h, d, m, u0, seed, steps, every);
                    let (nu, se) = measure::shear_nu(&s, k, 2.0 * every as f64, floor);
                    println!(
                        "{mn},{d},{w},{h},{m},{k:.6},{seed},{steps},{nu:.5},{se:.5},{nu_th:.5}"
                    );
                    eprintln!(
                        "  {mn} d {d} h {h} seed {seed}: nu {nu:.4} (theory {nu_th:.4}) {:.1} s",
                        t0.elapsed().as_secs_f64()
                    );
                    if !series.is_empty() && seed == 1 {
                        for i in 0..s.t.len() {
                            out += &format!("{mn},{d},{h},{},{:.6}\n", s.t[i], s.amp()[i]);
                        }
                    }
                }
            }
        }
    }
    if !series.is_empty() {
        std::fs::write(&series, format!("model,d,height,t,amp\n{out}")).unwrap();
    }
}

/// Galilean factor: phase speed of a transverse wave in a uniform flow,
/// divided by the flow speed.
fn advect(args: &[String]) {
    let models: Vec<String> = list(args, "--model", "fhp3,ternary");
    let ds: Vec<f64> = list(args, "--d", "0.1,0.2,0.28,0.35,0.42");
    let us: Vec<f64> = list(args, "--u", "0.1,0.2");
    let w = arg(args, "--width", 512usize);
    let h = arg(args, "--height", 128usize);
    let a0 = arg(args, "--a0", 0.05f64);
    let steps = arg(args, "--steps", 4000usize);
    let seeds = arg(args, "--seeds", 3u32);
    let k = 2.0 * std::f64::consts::PI / w as f64;
    println!("model,d,u_set,u_measured,seed,steps,c,c_se,g,g_se,g_theory");
    for mn in &models {
        let md = model(mn);
        for &d in &ds {
            if md == Model::Fhp3 && d >= 0.5 {
                continue;
            }
            let gth = ternary::g_theory(md.capacity(), md.capacity(), d);
            for &u in &us {
                for seed in 1..=seeds {
                    let (s, ux) = measure::advect_series(md, w, h, d, u, 1, a0, seed, steps, 20);
                    let (c, se) = measure::phase_speed(&s, k);
                    println!(
                        "{mn},{d},{u},{ux:.5},{seed},{steps},{c:.5},{se:.5},{:.4},{:.4},{gth:.4}",
                        c / ux,
                        se / ux
                    );
                    eprintln!(
                        "  {mn} d {d} u {u} seed {seed}: g {:.4} (theory {gth:.4})",
                        c / ux
                    );
                }
            }
        }
    }
}

/// Block-averaged velocity noise at rest.
fn noise(args: &[String]) {
    let models: Vec<String> = list(args, "--model", "fhp3,ternary");
    let ds: Vec<f64> = list(args, "--d", "0.1,0.2,0.28,0.35,0.42");
    let blocks: Vec<usize> = list(args, "--blocks", "1,2,4,8,16,32");
    let w = arg(args, "--width", 512usize);
    let h = arg(args, "--height", 512usize);
    println!("model,d,block,std_ux,std_uy,theory");
    for mn in &models {
        let md = model(mn);
        for &d in &ds {
            if md == Model::Fhp3 && d >= 0.5 {
                continue;
            }
            for (b, sx, sy) in measure::rest_noise(md, w, h, d, 1, &blocks, 200, 10, 50) {
                println!(
                    "{mn},{d},{b},{sx:.6},{sy:.6},{:.6}",
                    measure::noise_theory(md, d, b)
                );
            }
        }
    }
}

fn write_png(path: &str, rgba: &[u8], w: usize, h: usize, scale: usize) {
    let (sw, sh) = (w * scale, h * scale);
    let mut img = vec![0u8; sw * sh * 4];
    for y in 0..sh {
        for x in 0..sw {
            let s = ((y / scale) * w + x / scale) * 4;
            img[(y * sw + x) * 4..][..4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    let f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut e = png::Encoder::new(f, sw as u32, sh as u32);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&img).unwrap();
}

/// Flow past a plate, `fhp run`'s setup (plate H/8 rows at W/6, one row off
/// centre, inlet 32 columns, randomize outlet 64 columns refreshed every 32
/// steps, y periodic) for either model on the CPU, at density --d and inflow
/// --u. Time-averaged vorticity PNGs, plus a probe of block u_y behind the
/// plate for the shedding frequency.
fn street(args: &[String]) {
    use fhp::coarse::{BlockSums, Field};
    use fhp::ternary::engine::{self, Ternary};
    use fhp::{bitsliced::Bitsliced, Obstacle, Setup};
    let md = model(&arg(args, "--model", "ternary".to_string()));
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 512usize);
    let d = arg(args, "--d", 0.28f64);
    let u = arg(args, "--u", 0.2f64);
    let steps = arg(args, "--steps", 40000usize);
    let g_used = arg(
        args,
        "--g",
        ternary::g_theory(md.capacity(), md.capacity(), d),
    );
    let nu_used = arg(
        args,
        "--nu",
        match md {
            Model::Fhp3 => ternary::nu_fhp3(d),
            Model::Ternary(r) => ternary::nu_ternary(d, r),
        },
    );
    let (bw, bh) = (16usize, 16usize);
    let every = 5usize;
    let frame = 50usize;
    let alpha = arg(args, "--alpha", 0.5f32);
    let out = arg(args, "--out", "out".to_string());
    let pngs: Vec<usize> = list(args, "--png", "10000,20000,30000,40000");
    std::fs::create_dir_all(&out).unwrap();
    let mut s = Setup::street(w, h, u);
    s.density = d;
    let b = s.params.boundary.unwrap();
    s.params.boundary = Some(fhp::boundary::Boundary::new(
        d, u, b.inlet, b.outlet, b.kind,
    ));
    let rows = s.plate_rows();
    let dlen = rows as f64 * 3f64.sqrt() / 2.0;
    let re = u * dlen * g_used / nu_used;
    let vmax = arg(args, "--vmax", (4.0 * u / dlen) as f32);
    println!("{} {w}x{h} d {d} U {u} (Mach {:.3}) plate {rows} rows ({dlen:.1} across), g {g_used:.4}, nu {nu_used:.4}, Re = U D g / nu = {re:.1}", md.name(), Setup::mach(u));
    enum Eng {
        F(Box<Bitsliced>),
        T(Box<Ternary>),
    }
    let mut eng = match md {
        Model::Fhp3 => Eng::F(Box::new(Bitsliced::from_setup(&s))),
        Model::Ternary(rule) => {
            let mean = ternary::equilibrium(2, 2, d, u, 0.0);
            let obst = s.obstacle;
            let cells: Vec<u16> = (0..w * h)
                .map(|i| {
                    if obst.contains(i % w, i / w) {
                        engine::OBST
                    } else {
                        engine::sample_equilibrium(&mean, 1, i as u32)
                    }
                })
                .collect();
            Eng::T(Box::new(Ternary::new(
                w,
                h,
                cells,
                engine::Params {
                    seed: 1,
                    rule,
                    boundary: Some(engine::Boundary::new(d, u)),
                },
            )))
        }
    };
    let Obstacle::Plate { x: px, y: py, .. } = s.obstacle else {
        unreachable!()
    };
    // Probe: the block 2 plate lengths downstream, on the plate's row.
    let probe = (
        (py / bh) * (w / bw) + (px + (2.0 * dlen) as usize) / bw,
        ((py / bh) * (w / bw)) + (px + (4.0 * dlen) as usize) / bw,
    );
    let mut field = Field::new(w / bw, h / bh, bw, bh);
    let mut acc: Option<BlockSums> = None;
    let mut trace = String::from("step,uy_probe2,uy_probe4,mean_ux\n");
    let t0 = Instant::now();
    let mut done = 0;
    while done < steps {
        match &mut eng {
            Eng::F(b) => b.run(every),
            Eng::T(t) => t.run(every),
        }
        done += every;
        let x = match &eng {
            Eng::F(b) => b.block_sums(bw, bh),
            Eng::T(t) => t.block_sums(bw, bh),
        };
        match acc.as_mut() {
            Some(a) => a.add(&x),
            None => acc = Some(x),
        }
        if done % frame != 0 {
            continue;
        }
        let sums = acc.take().unwrap();
        // Instantaneous (frame-summed) probe velocity.
        let pv = |b: usize| sums.jy[b] as f64 * 3f64.sqrt() / 2.0 / sums.mass[b].max(1) as f64;
        field.add(&sums, alpha);
        trace += &format!(
            "{done},{:.5},{:.5},{:.5}\n",
            pv(probe.0),
            pv(probe.1),
            field.mean_ux()
        );
        if done % 2000 == 0 {
            eprintln!(
                "step {done} mean ux {:.4} ({:.0} s)",
                field.mean_ux(),
                t0.elapsed().as_secs_f64()
            );
        }
        if pngs.contains(&done) {
            write_png(
                &format!("{out}/vorticity-{done:06}.png"),
                &field.vorticity_rgba(vmax),
                field.nx,
                field.ny,
                2,
            );
        }
    }
    // Shedding frequency: the peak of the probe's periodogram over the
    // second half of the run (mean removed), scanned finely in frequency.
    let rows: Vec<(f64, f64, f64)> = trace
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        })
        .filter(|r| r.0 > steps as f64 / 2.0)
        .collect();
    std::fs::write(format!("{out}/probe.csv"), trace).unwrap();
    let span = rows.last().unwrap().0 - rows[0].0;
    for (name, col) in [("probe 2D", 1usize), ("probe 4D", 2)] {
        let v: Vec<f64> = rows
            .iter()
            .map(|r| if col == 1 { r.1 } else { r.2 })
            .collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let (mut best, mut bf) = (0.0, 0.0);
        let mut f = 2.0 / span;
        while f < 0.002 {
            let (mut c, mut sn) = (0.0, 0.0);
            for (r, x) in rows.iter().zip(&v) {
                c += (x - m) * (2.0 * std::f64::consts::PI * f * r.0).cos();
                sn += (x - m) * (2.0 * std::f64::consts::PI * f * r.0).sin();
            }
            if c * c + sn * sn > best {
                best = c * c + sn * sn;
                bf = f;
            }
            f += 0.02 / span;
        }
        let rms = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        println!(
            "{name}: rms u_y {rms:.4}, peak frequency {bf:.3e} per step (period {:.0} steps, {:.1} periods in the window), St = f D / (g U) = {:.3}",
            1.0 / bf,
            span * bf,
            bf * dlen / (g_used * u)
        );
    }
    let ux = field.mean_ux() as f64;
    println!(
        "mean u_x {ux:.4}, Re with the mean speed = {:.1}; {} steps in {:.0} s",
        ux * dlen * g_used / nu_used,
        steps,
        t0.elapsed().as_secs_f64()
    );
}

fn read_csv(path: &str) -> Vec<std::collections::HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines = text.lines();
    let Some(head) = lines.next() else {
        return vec![];
    };
    let keys: Vec<&str> = head.split(',').collect();
    lines
        .filter(|l| !l.is_empty())
        .map(|l| {
            keys.iter()
                .zip(l.split(','))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .collect()
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let sd = if v.len() > 1 {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        f64::NAN
    };
    (m, sd)
}

/// Tables (markdown, stdout) and aggregated CSVs for the plots, from the
/// CSVs in --dir.
fn summarize(args: &[String]) {
    let dir = arg(
        args,
        "--dir",
        "docs/runs/2026-10-06-fhp-ternary".to_string(),
    );
    let f =
        |r: &std::collections::HashMap<String, String>, k: &str| -> f64 { r[k].parse().unwrap() };
    let models = ["fhp3", "ternary", "ternary-optimal", "ternary-chiral"];
    let shear = read_csv(&format!("{dir}/shear.csv"));
    let advect = read_csv(&format!("{dir}/advect.csv"));
    let noise = read_csv(&format!("{dir}/noise.csv"));
    let mut ds: Vec<(String, f64)> = vec![];
    for r in shear.iter().chain(&advect) {
        let key = (r["model"].clone(), f(r, "d"));
        if !ds.contains(&key) {
            ds.push(key);
        }
    }
    ds.sort_by(|a, b| {
        (models.iter().position(|m| *m == a.0), a.1)
            .partial_cmp(&(models.iter().position(|m| *m == b.0), b.1))
            .unwrap()
    });
    let pick = |rows: &[std::collections::HashMap<String, String>],
                m: &str,
                d: f64,
                key: &str,
                val: &str,
                want: f64|
     -> Vec<f64> {
        rows.iter()
            .filter(|r| {
                r["model"] == m && (f(r, "d") - d).abs() < 1e-9 && (f(r, key) - want).abs() < 1e-9
            })
            .map(|r| f(r, val))
            .collect()
    };
    let fmt = |v: &[f64]| -> String {
        if v.is_empty() {
            return "".into();
        }
        let (m, sd) = mean_sd(v);
        if sd.is_nan() {
            format!("{m:.4}")
        } else {
            format!("{m:.4} ± {sd:.4}")
        }
    };
    // 4096 x 1024 street: plate 128 rows, D = 128 sqrt3/2, U = 0.2.
    let dplate = 128.0 * 3f64.sqrt() / 2.0;
    let mut vcsv = String::from("model,d,nu64,nu64_sd,nu128,nu128_sd,nu_theory,g01,g01_sd,g02,g_theory,g_over_nu,g_over_nu_theory\n");
    println!("### Viscosity (shear-wave decay, width 1024, u0 = 0.1, mean ± sd over 3 seeds at H = 64 and 128, 2 at H = 256)\n");
    println!("| model | d | nu, H=64 | nu, H=128 | nu, H=256 | nu, Boltzmann theory |\n|---|---|---|---|---|---|");
    for (m, d) in &ds {
        let th = shear
            .iter()
            .find(|r| &r["model"] == m && (f(r, "d") - d).abs() < 1e-9)
            .map(|r| f(r, "nu_theory"));
        let Some(th) = th else { continue };
        println!(
            "| {m} | {d} | {} | {} | {} | {th:.4} |",
            fmt(&pick(&shear, m, *d, "height", "nu", 64.0)),
            fmt(&pick(&shear, m, *d, "height", "nu", 128.0)),
            fmt(&pick(&shear, m, *d, "height", "nu", 256.0))
        );
    }
    println!("\n### Galilean factor (phase speed of a transverse wave / flow speed, width 512 x 128, mean ± sd over 3 seeds)\n");
    println!("| model | d | g, U=0.1 | g, U=0.2 | g, theory |\n|---|---|---|---|---|");
    for (m, d) in &ds {
        let th = advect
            .iter()
            .find(|r| &r["model"] == m && (f(r, "d") - d).abs() < 1e-9)
            .map(|r| f(r, "g_theory"));
        let Some(th) = th else { continue };
        println!(
            "| {m} | {d} | {} | {} | {th:.4} |",
            fmt(&pick(&advect, m, *d, "u_set", "g", 0.1)),
            fmt(&pick(&advect, m, *d, "u_set", "g", 0.2))
        );
    }
    println!("\n### Reynolds number per lattice size: g/nu (measured g at U=0.1, nu at H=128 and H=256) and Re = U D g/nu for the 4096 x 1024 street (U = 0.2, D = {dplate:.1}) with nu at H=128\n");
    println!("| model | d | g | nu, H=128 | g/nu, H=128 | g/nu, H=256 | g/nu, theory | Re at 4096 x 1024 |\n|---|---|---|---|---|---|---|---|");
    for (m, d) in &ds {
        let g = pick(&advect, m, *d, "u_set", "g", 0.1);
        let g2 = pick(&advect, m, *d, "u_set", "g", 0.2);
        let nu = pick(&shear, m, *d, "height", "nu", 128.0);
        let nu64 = pick(&shear, m, *d, "height", "nu", 64.0);
        if g.is_empty() || nu.is_empty() {
            continue;
        }
        let gth = advect
            .iter()
            .find(|r| &r["model"] == m && (f(r, "d") - d).abs() < 1e-9)
            .map(|r| f(r, "g_theory"))
            .unwrap();
        let nuth = shear
            .iter()
            .find(|r| &r["model"] == m && (f(r, "d") - d).abs() < 1e-9)
            .map(|r| f(r, "nu_theory"))
            .unwrap();
        let (gm, gs) = mean_sd(&g);
        let (nm, ns) = mean_sd(&nu);
        let (n6, n6s) = mean_sd(&nu64);
        let (g2m, _) = mean_sd(&g2);
        let nu256 = pick(&shear, m, *d, "height", "nu", 256.0);
        let c256 = if nu256.is_empty() {
            String::new()
        } else {
            format!("{:.2}", gm / mean_sd(&nu256).0)
        };
        println!(
            "| {m} | {d} | {gm:.4} | {nm:.4} | {:.2} | {c256} | {:.2} | {:.0} |",
            gm / nm,
            gth / nuth,
            0.2 * dplate * gm / nm
        );
        vcsv += &format!(
            "{m},{d},{n6},{n6s},{nm},{ns},{nuth},{gm},{gs},{g2m},{gth},{},{}\n",
            gm / nm,
            gth / nuth
        );
    }
    std::fs::write(format!("{dir}/summary.csv"), vcsv).unwrap();
    println!("\n### Noise at rest: standard deviation of block-averaged u_x (512 x 512, 10 snapshots), b x b blocks\n");
    let blocks: Vec<usize> = {
        let mut b: Vec<usize> = noise.iter().map(|r| f(r, "block") as usize).collect();
        b.sort();
        b.dedup();
        b
    };
    let head: Vec<String> = blocks.iter().map(|b| format!("b={b}")).collect();
    println!(
        "| model | d | {} | theory b=16 |\n|---|---|{}---|",
        head.join(" | "),
        "---|".repeat(blocks.len())
    );
    let mut keys: Vec<(String, f64)> = vec![];
    for r in &noise {
        let k = (r["model"].clone(), f(r, "d"));
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    for (m, d) in keys {
        let cells: Vec<String> = blocks
            .iter()
            .map(|&b| {
                pick(&noise, &m, d, "block", "std_ux", b as f64)
                    .first()
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_default()
            })
            .collect();
        let th = pick(&noise, &m, d, "block", "theory", 16.0);
        println!(
            "| {m} | {d} | {} | {:.4} |",
            cells.join(" | "),
            th.first().copied().unwrap_or(f64::NAN)
        );
    }
}

/// Search variable orders for the smallest circuit of rule Chiral (both
/// encodings, both forms): every order that keeps each channel's two planes
/// adjacent, over channel orders, plane order within the pairs, and the
/// positions of obstacle and chirality; then pairwise swaps from the best.
/// Writes src/ternary/collide_gen.rs and src/shaders/ternary_collide_gen.wgsl
/// unless --dry.
fn gen(args: &[String]) {
    use fhp::ternary::synth::{self, Encoding, NIN};
    use rayon::prelude::*;
    let t = ternary::Table::new();
    let budget = arg(args, "--budget", 3000usize);
    let t0 = Instant::now();
    let mut best: Option<synth::Circuit> = None;
    for enc in [Encoding::Binary, Encoding::Thermometer] {
        for flip in [false, true] {
            let tts = synth::truth_tables(&t, enc, flip);
            let one = std::time::Instant::now();
            let c = synth::synthesize(&tts, enc, &(0..NIN).collect::<Vec<_>>(), flip);
            eprintln!("{enc:?} flip {flip}: natural order {} gates ({:.2} s)", c.gate_count(), one.elapsed().as_secs_f64());
            // Seeds: channel orders (sampled), pairs kept adjacent, obstacle and chirality first or last.
            let mut cands: Vec<Vec<usize>> = vec![];
            let chans = fhp::synth::permutations(7);
            let mut k = 0u32;
            for (ci, ch) in chans.iter().enumerate() {
                if ci % (chans.len() / (budget / 8).max(1)).max(1) != 0 {
                    continue;
                }
                for head in [[14usize, 15], [15, 14]] {
                    for swap in [false, true] {
                        k += 1;
                        let mut o = head.to_vec();
                        for &c in ch {
                            if swap ^ (fhp::lowbias32(k ^ c as u32) & 1 == 1) {
                                o.extend([2 * c + 1, 2 * c]);
                            } else {
                                o.extend([2 * c, 2 * c + 1]);
                            }
                        }
                        cands.push(o.clone());
                        o.rotate_left(1);
                        cands.push(o);
                    }
                }
            }
            let mut res: Vec<(usize, Vec<usize>)> = cands.par_iter().map(|o| (synth::synthesize(&tts, enc, o, flip).gate_count(), o.clone())).collect();
            res.sort();
            let (mut bc, mut bo) = res[0].clone();
            eprintln!("  {} seeded orders: best {bc} gates {bo:?} ({:.0} s)", res.len(), t0.elapsed().as_secs_f64());
            // Improve by pairwise swaps until no swap helps.
            loop {
                let swaps: Vec<(usize, usize)> = (0..NIN).flat_map(|a| (a + 1..NIN).map(move |b| (a, b))).collect();
                let r = swaps
                    .par_iter()
                    .map(|&(a, b)| {
                        let mut o = bo.clone();
                        o.swap(a, b);
                        (synth::synthesize(&tts, enc, &o, flip).gate_count(), o)
                    })
                    .min()
                    .unwrap();
                if r.0 >= bc {
                    break;
                }
                (bc, bo) = r;
                eprintln!("  swap: {bc} gates ({:.0} s)", t0.elapsed().as_secs_f64());
            }
            let c = synth::synthesize(&tts, enc, &bo, flip);
            synth::verify(&c, &t).expect("circuit matches the table");
            println!("{enc:?} {} form: {} gates {:?}, order {:?}", if flip { "flip" } else { "direct" }, c.gate_count(), c.counts(), c.order);
            if best.as_ref().is_none_or(|b| c.gate_count() < b.gate_count()) {
                best = Some(c);
            }
        }
    }
    let c = best.unwrap();
    println!("best: {:?} {} form, {} gates {:?}, order {:?} ({:.0} s)", c.enc, if c.flip { "flip" } else { "direct" }, c.gate_count(), c.counts(), c.order, t0.elapsed().as_secs_f64());
    if !args.iter().any(|a| a == "--dry") {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        std::fs::write(format!("{dir}/ternary/collide_gen.rs"), c.to_rust()).unwrap();
        std::fs::write(format!("{dir}/shaders/ternary_collide_gen.wgsl"), c.to_wgsl()).unwrap();
    }
}
