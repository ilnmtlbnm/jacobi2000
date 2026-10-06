//! Signed ternary FHP measurements, with the setups of `ternary`:
//! `theory`, `shear`, `advect`, `noise`, `street`, `bench`. CSV on stdout.
//! Configurations are the product of --rule, --block, --every and --n0.

#![allow(clippy::needless_range_loop)]

use fhp::signed::{self, measure, Params, Signed};
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
    arg(args, name, default.to_string()).split(',').filter_map(|x| x.parse().ok()).collect()
}

fn rule_name(r: Rule) -> String {
    match r {
        Rule::Mix(k) => format!("mix{k}"),
        _ => format!("{r:?}").to_lowercase(),
    }
}

/// Every configuration asked for, with its name.
fn configs(args: &[String]) -> Vec<(String, Params)> {
    let rules: Vec<String> = list(args, "--rule", "optimal");
    let mut v = vec![];
    for r in &rules {
        let rule = Rule::parse(r).expect("rule");
        for b in list::<usize>(args, "--block", "8") {
            for n in list::<u32>(args, "--every", "1") {
                for n0 in list::<f64>(args, "--n0", "4") {
                    let name = format!("signed-{}-B{b}-N{n}-n{n0}", rule_name(rule));
                    v.push((name, Params { seed: 1, rule, block: b, every: n, n0, boundary: None }));
                }
            }
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if std::env::var("RHOONLY").is_ok() {
        signed::RHOONLY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("COLUMNS").is_ok() {
        signed::COLUMNS.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("NOKICK").is_ok() {
        signed::NOKICK.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let rest = &args[1.min(args.len())..];
    match args.first().map(|s| s.as_str()) {
        Some("theory") => theory(),
        Some("shear") => shear(rest),
        Some("advect") => advect(rest),
        Some("noise") => noise(rest),
        Some("street") => street(rest),
        Some("bench") => bench(rest),
        Some("debug") => debug(rest),
        Some("shear-delta") => shear_delta(rest),
        Some("stack") => stack(rest),
        Some("summarize") => summarize(rest),
        Some("bands") => bands(rest),
        Some("bandstat") => bandstat(rest),
        Some("mirror") => mirror_check(rest),
        Some("vortex") => vortex(rest),
        Some("handed") => handed(rest),
        _ => {
            eprintln!("usage: signed theory | shear | advect | noise | street | bench");
            std::process::exit(2);
        }
    }
}

/// Boltzmann viscosity of the half-filled deviation gas per rule, and the
/// noise of independent half-filled deviations per n0.
fn theory() {
    println!("rule,nu_half_filled,g,g_over_nu");
    for r in [Rule::Other, Rule::Uniform, Rule::Optimal, Rule::Chiral] {
        let nu = ternary::nu_ternary(1.0, r);
        println!("{},{nu:.5},1,{:.2}", rule_name(r), 1.0 / nu);
    }
    eprintln!("n0,noise_b16,relative_noise_per_channel");
    for n0 in [1.0, 2.0, 4.0, 8.0] {
        eprintln!("{n0},{:.5},{:.4}", measure::noise_theory(n0, 16), (2.0f64 / 3.0).sqrt() / n0);
    }
}

fn shear(args: &[String]) {
    let hs: Vec<usize> = list(args, "--h", "64,128");
    let w = arg(args, "--width", 1024usize);
    let u0 = arg(args, "--u0", 0.1f64);
    let seeds = arg(args, "--seeds", 3u32);
    let floor = arg(args, "--floor", 0.4f64);
    let nu_guess = arg(args, "--nu-guess", 0.0f64);
    println!("model,rule,block,every,n0,width,height,k,seed,steps,nu,nu_se,nu_theory,saturated_per_site_step,kicks_per_site_step");
    for (name, p) in configs(args) {
        let nu_th = ternary::nu_ternary(1.0, p.rule);
        for &h in &hs {
            let k = 2.0 * std::f64::consts::PI / (h as f64 * 3f64.sqrt() / 2.0);
            let g = if nu_guess > 0.0 { nu_guess } else { 2.5 * nu_th };
            let steps = (((1.0 / floor).ln() * 1.3 / (g.max(0.005) * k * k)) as usize).clamp(2000, 200_000);
            let every = (steps / 200).max(1);
            for seed in 1..=seeds {
                let t0 = Instant::now();
                let (s, sat, kicks) = shear_run(p, w, h, u0, seed, steps, every);
                let (nu, se) = measure::shear_nu(&s, k, 2.0 * every as f64, floor);
                let ss = (w * h * steps) as f64;
                println!(
                    "{name},{},{},{},{},{w},{h},{k:.6},{seed},{steps},{nu:.5},{se:.5},{nu_th:.5},{:.3e},{:.3e}",
                    rule_name(p.rule),
                    p.block,
                    p.every,
                    p.n0,
                    sat as f64 / ss,
                    kicks as f64 / ss
                );
                eprintln!("  {name} h {h} seed {seed}: nu {nu:.4} (theory {nu_th:.4}) {:.1} s", t0.elapsed().as_secs_f64());
            }
        }
    }
}

fn shear_run(p: Params, w: usize, h: usize, u0: f64, seed: u32, steps: usize, every: usize) -> (measure::Series, u64, u64) {
    let ky = 2.0 * std::f64::consts::PI / (h as f64 * 3f64.sqrt() / 2.0);
    let mut l = measure::lattice(w, h, p, seed, |_, y| (u0 * (ky * y).sin(), 0.0));
    let s = series(&mut l, steps, every, false, 0.0, ky);
    (s, l.saturated, l.kicks)
}

fn series(l: &mut Signed, steps: usize, every: usize, comp_y: bool, kx: f64, ky: f64) -> measure::Series {
    let mut s = measure::Series { t: vec![], re: vec![], im: vec![] };
    let mut t = 0;
    loop {
        let (re, im) = measure::project(l, comp_y, kx, ky);
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

fn advect(args: &[String]) {
    let us: Vec<f64> = list(args, "--u", "0.1,0.2");
    let w = arg(args, "--width", 512usize);
    let h = arg(args, "--height", 128usize);
    let a0 = arg(args, "--a0", 0.05f64);
    let steps = arg(args, "--steps", 4000usize);
    let seeds = arg(args, "--seeds", 3u32);
    let k = 2.0 * std::f64::consts::PI / w as f64;
    println!("model,rule,block,every,n0,u_set,u_measured,seed,steps,c,c_se,g,g_se,g_theory,amp_end");
    for (name, p) in configs(args) {
        for &u in &us {
            for seed in 1..=seeds {
                let mut l = measure::lattice(w, h, p, seed, |x, _| (u, a0 * (k * x).sin()));
                let (ms, jx, _) = l.fields();
                let ux = jx.iter().sum::<f64>() / ms.iter().sum::<f64>();
                let s = series(&mut l, steps, 20, true, k, 0.0);
                let (c, se) = measure::phase_speed(&s, k);
                let amp = *s.amp().last().unwrap() / s.amp()[0];
                println!(
                    "{name},{},{},{},{},{u},{ux:.5},{seed},{steps},{c:.5},{se:.5},{:.4},{:.4},1,{amp:.4}",
                    rule_name(p.rule),
                    p.block,
                    p.every,
                    p.n0,
                    c / ux,
                    se / ux
                );
                eprintln!("  {name} u {u} seed {seed}: g {:.4}", c / ux);
            }
        }
    }
}

fn noise(args: &[String]) {
    let blocks: Vec<usize> = list(args, "--blocks", "1,2,4,8,16,32");
    let w = arg(args, "--width", 512usize);
    let h = arg(args, "--height", 512usize);
    println!("model,rule,block,every,n0,b,std_ux,std_uy,theory");
    for (name, p) in configs(args) {
        for (b, sx, sy) in measure::rest_noise(p, w, h, 1, &blocks, 200, 10, 50) {
            println!(
                "{name},{},{},{},{},{b},{sx:.6},{sy:.6},{:.6}",
                rule_name(p.rule),
                p.block,
                p.every,
                p.n0,
                measure::noise_theory(p.n0, b)
            );
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

/// Flow past a plate with `ternary street`'s setup (plate H/8 rows at W/6,
/// one row off centre, inlet 32 and outlet 64 columns refreshed every 32
/// steps to the inflow equilibrium, y periodic), the same block averaging,
/// vorticity scale and probes.
fn street(args: &[String]) {
    use fhp::coarse::Field;
    use fhp::{Obstacle, Setup};
    let (_, p0) = configs(args).remove(0);
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 512usize);
    let u = arg(args, "--u", 0.2f64);
    let steps = arg(args, "--steps", 40000usize);
    let g_used = arg(args, "--g", 1.0f64);
    let nu_used = arg(args, "--nu", ternary::nu_ternary(1.0, p0.rule));
    let (bw, bh) = (16usize, 16usize);
    let (every, frame) = (5usize, 50usize);
    let alpha = arg(args, "--alpha", 0.5f64);
    let out = arg(args, "--out", "out".to_string());
    let pngs: Vec<usize> = list(args, "--png", "10000,20000,30000,40000");
    std::fs::create_dir_all(&out).unwrap();
    let s = Setup::street(w, h, u);
    let rows = s.plate_rows();
    let dlen = rows as f64 * 3f64.sqrt() / 2.0;
    let re = u * dlen * g_used / nu_used;
    let vmax = arg(args, "--vmax", (4.0 * u / dlen) as f32);
    let rho0 = 7.0 * p0.n0;
    let p = Params { boundary: Some(signed::Boundary { rho: rho0, u, inlet: 32, outlet: 64 }), ..p0 };
    println!(
        "signed rule {} B {} N {} n0 {} {w}x{h} U {u} (Mach {:.3}) plate {rows} rows ({dlen:.1} across), g {g_used:.4}, nu {nu_used:.4}, Re = U D g / nu = {re:.1}",
        rule_name(p.rule),
        p.block,
        p.every,
        p.n0,
        Setup::mach(u)
    );
    let obst = s.obstacle;
    let mut eng = Signed::sample(w, h, p, |x, y| obst.contains(x, y), |_, _| (rho0, u, 0.0), 1);
    let Obstacle::Plate { x: px, y: py, .. } = s.obstacle else { unreachable!() };
    let nbx = w / bw;
    let probe = ((py / bh) * nbx + (px + (2.0 * dlen) as usize) / bw, (py / bh) * nbx + (px + (4.0 * dlen) as usize) / bw);
    let mut field = Field::new(w / bw, h / bh, bw, bh);
    let mut acc: Option<signed::BlockSums> = None;
    let mut trace = String::from("step,uy_probe2,uy_probe4,mean_ux\n");
    let t0 = Instant::now();
    let mut done = 0;
    while done < steps {
        eng.run(every);
        done += every;
        let x = eng.block_sums(bw, bh);
        match acc.as_mut() {
            Some(a) => a.add(&x),
            None => acc = Some(x),
        }
        if done % frame != 0 {
            continue;
        }
        let sums = acc.take().unwrap();
        let pv = |b: usize| sums.jy[b] / sums.mass[b];
        let a = if field.samples == 0 { 1.0 } else { alpha };
        for b in 0..field.nx * field.ny {
            let (ux, uy) = (sums.jx[b] / sums.mass[b], sums.jy[b] / sums.mass[b]);
            field.ux[b] += (a * (ux - field.ux[b] as f64)) as f32;
            field.uy[b] += (a * (uy - field.uy[b] as f64)) as f32;
            field.solid[b] = sums.fluid[b] < (bw * bh * frame / every) as u32;
        }
        field.samples += 1;
        trace += &format!("{done},{:.5},{:.5},{:.5}\n", pv(probe.0), pv(probe.1), field.mean_ux());
        if done % 2000 == 0 {
            let (m, jx, jy) = eng.totals();
            eprintln!(
                "step {done} mean ux {:.4} total mass {m:.1} momentum ({jx:.1}, {jy:.1}) saturated/site/step {:.2e} ({:.0} s)",
                field.mean_ux(),
                eng.saturated as f64 / (w * h * done) as f64,
                t0.elapsed().as_secs_f64()
            );
            if !field.mean_ux().is_finite() {
                println!("diverged at step {done}");
                break;
            }
        }
        if pngs.contains(&done) {
            write_png(&format!("{out}/vorticity-{done:06}.png"), &field.vorticity_rgba(vmax), field.nx, field.ny, 2);
        }
    }
    let rows: Vec<(f64, f64, f64)> = trace
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            (v[0], v[1], v[2])
        })
        .filter(|r| r.0 > steps as f64 / 2.0)
        .collect();
    std::fs::write(format!("{out}/probe.csv"), &trace).unwrap();
    if rows.len() > 10 {
        let span = rows.last().unwrap().0 - rows[0].0;
        for (name, col) in [("probe 2D", 1usize), ("probe 4D", 2)] {
            let v: Vec<f64> = rows.iter().map(|r| if col == 1 { r.1 } else { r.2 }).collect();
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
    }
    let ux = field.mean_ux() as f64;
    println!(
        "mean u_x {ux:.4}, Re with the mean speed = {:.1}; saturated per site-step {:.3e}, kicks per site-step {:.3e}; {} steps in {:.0} s",
        ux * dlen * g_used / nu_used,
        eng.saturated as f64 / (w * h * done) as f64,
        eng.kicks as f64 / (w * h * done) as f64,
        done,
        t0.elapsed().as_secs_f64()
    );
}

fn bench(args: &[String]) {
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 512usize);
    let steps = arg(args, "--steps", 50usize);
    for (name, p) in configs(args) {
        for threads in [1, rayon::current_num_threads()] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut l = measure::lattice(w, h, p, 1, |x, _| (0.2, 0.02 * (x / 50.0).sin()));
                l.run(2);
                let t0 = Instant::now();
                l.run(steps);
                let secs = t0.elapsed().as_secs_f64();
                println!("{name:<34} {threads} thr {w}x{h} {steps:>5} steps {secs:>8.3} s {:>8.1} M site-updates/s", (w * h * steps) as f64 / secs / 1e6);
            });
        }
        if args.iter().any(|a| a == "--gpu") {
            let gsteps = arg(args, "--gpu-steps", 1000usize);
            let l = measure::lattice(w, h, p, 1, |x, _| (0.2, 0.02 * (x / 50.0).sin()));
            let mut g = pollster::block_on(signed::gpu::Gpu::from_cpu(&l)).expect("GPU adapter");
            g.run(50);
            g.finish();
            let t0 = Instant::now();
            g.run(gsteps);
            g.finish();
            let secs = t0.elapsed().as_secs_f64();
            println!("{name:<34} gpu {} {w}x{h} {gsteps:>5} steps {secs:>8.3} s {:>8.1} M site-updates/s", g.adapter, (w * h * gsteps) as f64 / secs / 1e6);
        }
    }
}

/// Per-step diagnostics of the reference at rest: block density and speed
/// extremes, reservoir size, saturations and kicks.
fn debug(args: &[String]) {
    let (_, p) = configs(args).remove(0);
    let w = arg(args, "--width", 256usize);
    let h = arg(args, "--height", 64usize);
    let steps = arg(args, "--steps", 200usize);
    let mut l = measure::lattice(w, h, p, 1, |_, _| (0.0, 0.0));
    println!("step,rho_min,rho_max,u_max,res_max,sat,kicks");
    let (mut s0, mut k0) = (0, 0);
    for t in 0..steps {
        l.step();
        let q = signed::ONE as f64;
        let rmin = l.nodes.iter().map(|v| v[0] as f64 / q).fold(f64::INFINITY, f64::min);
        let rmax = l.nodes.iter().map(|v| v[0] as f64 / q).fold(0.0, f64::max);
        let umax = l.nodes.iter().map(|v| (v[1] as f64 / 2.0).hypot(v[2] as f64 * 3f64.sqrt() / 2.0) / v[0] as f64).fold(0.0, f64::max);
        let res = l.res.iter().flatten().fold(0.0f64, |a, &b| a.max((b as f64 / q).abs()));
        println!("{t},{rmin:.3},{rmax:.3},{umax:.4},{res:.2},{},{}", l.saturated - s0, l.kicks - k0);
        (s0, k0) = (l.saturated, l.kicks);
    }
}

/// Shear wave held entirely in the deviations on a uniform reference at
/// rest (what re-estimation then moves into the reference).
fn shear_delta(args: &[String]) {
    let h = arg(args, "--h", 64usize);
    let w = arg(args, "--width", 256usize);
    let u0 = arg(args, "--u0", 0.02f64);
    let steps = arg(args, "--steps", 6000usize);
    let k = 2.0 * std::f64::consts::PI / (h as f64 * 3f64.sqrt() / 2.0);
    println!("model,nu,sat,kicks");
    for (name, p) in configs(args) {
        let rho0 = 7.0 * p.n0;
        let mut l = Signed::new(w, h, vec![signed::ZERO; w * h], p);
        for i in 0..w * h {
            let (_, py) = measure::position(i % w, i / w);
            let ux = u0 * (k * py).sin();
            let mut code = 0u16;
            for c in 0..7 {
                let m = (rho0 * signed::velocity(c).0 * ux / 3.0).clamp(-1.0, 1.0);
                let pr = ternary::ternary_probs(1.0 + m);
                let x = fhp::lowbias32(0x1234 ^ fhp::lowbias32((i * 8 + c) as u32)) as f64 / 4294967296.0;
                code += ((x >= pr[0]) as u16 + (x >= pr[0] + pr[1]) as u16) * ternary::POW3[c];
            }
            l.cells[i] = code;
        }
        let s = series(&mut l, steps, steps / 200, false, 0.0, k);
        let (nu, _) = measure::shear_nu(&s, k, 2.0 * (steps / 200) as f64, 0.4);
        if args.iter().any(|a| a == "--trace") {
            for j in (0..s.t.len()).step_by(20) {
                eprintln!("{name} t {} re {:.5} im {:.5}", s.t[j], s.re[j], s.im[j]);
            }
        }
        println!("{name},{nu:.5},{:.3e},{:.3e}", l.saturated as f64 / (w * h * steps * 7) as f64, l.kicks as f64 / (w * h * steps * 7) as f64);
    }
}

/// Stack RGBA PNGs of the same width vertically, 4 grey rows apart:
/// `stack out.png a.png b.png ...`.
fn stack(args: &[String]) {
    let mut rows: Vec<u8> = vec![];
    let mut width = 0;
    let mut height = 0;
    for (n, path) in args[1..].iter().enumerate() {
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let mut r = dec.read_info().unwrap();
        let mut buf = vec![0; r.output_buffer_size().unwrap()];
        let info = r.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgba);
        if n > 0 {
            assert_eq!(info.width as usize, width);
            rows.extend(std::iter::repeat_n([128u8, 128, 128, 255], width * 4).flatten());
            height += 4;
        }
        width = info.width as usize;
        rows.extend_from_slice(&buf[..info.buffer_size()]);
        height += info.height as usize;
    }
    write_png(&args[0], &rows, width, height, 1);
}

/// Mean and sample sd of a CSV column grouped by other columns:
/// `summarize file.csv value key1,key2`.
fn summarize(args: &[String]) {
    let text = std::fs::read_to_string(&args[0]).unwrap();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap_or_else(|| panic!("no column {n}"));
    let v = col(&args[1]);
    let keys: Vec<usize> = args[2].split(',').map(col).collect();
    let mut groups: Vec<(String, Vec<f64>)> = vec![];
    for l in lines.filter(|l| !l.is_empty()) {
        let f: Vec<&str> = l.split(',').collect();
        let k = keys.iter().map(|&i| f[i]).collect::<Vec<_>>().join(",");
        let x: f64 = f[v].parse().unwrap();
        match groups.iter_mut().find(|g| g.0 == k) {
            Some(g) => g.1.push(x),
            None => groups.push((k, vec![x])),
        }
    }
    println!("{},n,mean,sd", args[2]);
    for (k, x) in groups {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let sd = if x.len() > 1 { (x.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { f64::NAN };
        println!("{k},{},{m:.5},{sd:.5}", x.len());
    }
}

/// Wake structure indicators of a vorticity field (`nx` x `ny` display
/// blocks of `bw` x `bh` sites, y periodic) over the columns
/// [0.55, 0.85) nx, with D the plate length and `thr` the vorticity
/// threshold: (mean y-extent of the same-sign structures above `thr`,
/// weighted by their |vorticity| sum, in units of D; the largest y-extent
/// as a fraction of H; the y-spectrum power at wavelengths >= 4D over the
/// power at D/2 <= wavelength < 4D).
fn wake_stats(o: &[f32], nx: usize, ny: usize, bh: usize, d: f64, thr: f32) -> (f64, f64, f64) {
    let (x0, x1) = ((nx as f64 * 0.55) as usize, (nx as f64 * 0.85) as usize);
    let cw = x1 - x0;
    let dy = bh as f64 * 3f64.sqrt() / 2.0;
    let hp = ny as f64 * dy;
    let sign = |v: f32| if v > thr { 1i8 } else if v < -thr { -1 } else { 0 };
    let mut label = vec![usize::MAX; cw * ny];
    let (mut wsum, mut esum, mut emax) = (0.0f64, 0.0f64, 0.0f64);
    for start in 0..cw * ny {
        let sg = sign(o[(start / cw) * nx + x0 + start % cw]);
        if sg == 0 || label[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        label[start] = start;
        let mut rows = vec![false; ny];
        let mut weight = 0.0f64;
        while let Some(c) = stack.pop() {
            let (cx, cy) = (c % cw, c / cw);
            rows[cy] = true;
            weight += o[cy * nx + x0 + cx].abs() as f64;
            let mut nb = vec![((cy + 1) % ny) * cw + cx, ((cy + ny - 1) % ny) * cw + cx];
            if cx > 0 {
                nb.push(c - 1);
            }
            if cx + 1 < cw {
                nb.push(c + 1);
            }
            for n in nb {
                if label[n] == usize::MAX && sign(o[(n / cw) * nx + x0 + n % cw]) == sg {
                    label[n] = start;
                    stack.push(n);
                }
            }
        }
        let ext = rows.iter().filter(|&&r| r).count() as f64 * dy;
        wsum += weight;
        esum += weight * ext / d;
        emax = emax.max(ext / hp);
    }
    let (mut band, mut vort) = (0.0f64, 0.0f64);
    for x in x0..x1 {
        for k in 0..=ny / 2 {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for y in 0..ny {
                let a = 2.0 * std::f64::consts::PI * (k * y) as f64 / ny as f64;
                re += o[y * nx + x] as f64 * a.cos();
                im += o[y * nx + x] as f64 * a.sin();
            }
            let p = re * re + im * im;
            let lambda = if k == 0 { f64::INFINITY } else { hp / k as f64 };
            if lambda >= 4.0 * d {
                band += p;
            } else if lambda >= 0.5 * d {
                vort += p;
            }
        }
    }
    (if wsum > 0.0 { esum / wsum } else { 0.0 }, emax, band / vort.max(1e-30))
}

/// Peak frequency (per step) of a probe series (step, value) by a DFT scan.
fn peak_frequency(rows: &[(f64, f64)]) -> (f64, f64) {
    let span = rows.last().unwrap().0 - rows[0].0;
    let m = rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64;
    let (mut best, mut bf) = (0.0, 0.0);
    let mut f = 2.0 / span;
    while f < 0.002 {
        let (mut c, mut sn) = (0.0, 0.0);
        for r in rows {
            c += (r.1 - m) * (2.0 * std::f64::consts::PI * f * r.0).cos();
            sn += (r.1 - m) * (2.0 * std::f64::consts::PI * f * r.0).sin();
        }
        if c * c + sn * sn > best {
            best = c * c + sn * sn;
            bf = f;
        }
        f += 0.02 / span;
    }
    (bf, span * bf)
}

/// Diagnostic street on the GPU for the wake-band question
/// (docs/runs/2026-10-06-fhp-signed-bands.md): the page's street (plate at
/// W/6, inlet 32 and outlet 64 columns, U), model signed (reference block
/// --block, period --every) or ternary (the page's: chiral, density 0.6),
/// optional plate rows (--plate, default H/8) and free-slip walls at the
/// top and bottom rows (--walls, signed only). Display blocks of H/64,
/// sampled every 5 steps and folded every 50 with weight 0.5 (the page's
/// running average); vorticity PNGs and raw f32 fields at --png steps,
/// wake indicators every --stat-every steps, probes at 2D and 0.7 W.
fn bands(args: &[String]) {
    use fhp::coarse::{BlockSums, Field};
    use fhp::page;
    use fhp::{Obstacle, Setup};
    let model = arg(args, "--model", "signed".to_string());
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 512usize);
    let u = arg(args, "--u", 0.2f64);
    let steps = arg(args, "--steps", 125000usize);
    let rb = arg(args, "--block", 16usize);
    let every = arg(args, "--every", 4u32);
    let walls = args.iter().any(|a| a == "--walls");
    let out = arg(args, "--out", "out".to_string());
    let pngs: Vec<usize> = list(args, "--png", "20000,40000,60000,80000,100000,125000");
    let stat_every = arg(args, "--stat-every", 5000usize);
    let db = arg(args, "--display", h / 64);
    std::fs::create_dir_all(&out).unwrap();
    let mut s = Setup::street(w, h, u);
    let rows = arg(args, "--plate", h / 8);
    if let Obstacle::Plate { x, y, cols, .. } = s.obstacle {
        s.obstacle = Obstacle::Plate { x, y: arg(args, "--plate-y", y), rows, cols };
    }
    s.params.seed = arg(args, "--seed", s.params.seed);
    let dlen = rows as f64 * 3f64.sqrt() / 2.0;
    let (g, nu) = match model.as_str() {
        "ternary" => (page::TERNARY_G, page::TERNARY_NU),
        _ => page::signed_g_nu(rb),
    };
    let vmax = (4.0 * u / dlen) as f32;
    enum E {
        S(Box<signed::gpu::Gpu>),
        T(Box<ternary::gpu::Gpu>),
    }
    let mut e = if model == "ternary" {
        let s2 = Setup { density: page::TERNARY_DENSITY, ..s };
        let p = page::ternary_params(&s2);
        E::T(Box::new(pollster::block_on(ternary::gpu::Gpu::open(w, h, || page::ternary_cells(&s2), p.seed, ternary::gpu::Layout::Table, p.boundary)).unwrap()))
    } else {
        let p = Params { block: rb, every, ..page::signed_params(&s, rb) };
        if walls {
            signed::WALLS.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let rho = 7.0 * p.n0;
        let obst = s.obstacle;
        let (mut cells, nodes) = signed::initial(w, h, &p, |x, y| obst.contains(x, y), |_, _| (rho, u, 0.0), p.seed);
        if walls {
            for x in 0..w {
                cells[x] = signed::OBST | signed::WALL | signed::ZERO;
                cells[(h - 1) * w + x] = signed::OBST | signed::WALL | signed::ZERO;
            }
        }
        let nb = nodes.len();
        E::S(Box::new(pollster::block_on(signed::gpu::Gpu::new(w, h, p, || (cells, nodes, vec![[0; ternary::NCH]; nb]))).unwrap()))
    };
    println!(
        "bands model {model} B {rb} N {every} walls {walls} {w}x{h} U {u} plate {rows} rows at {:?} (D {dlen:.1}) seed {} g {g} nu {nu} Re {:.1} display block {db}",
        s.obstacle,
        s.params.seed,
        u * dlen * g / nu
    );
    let Obstacle::Plate { x: px, y: py, .. } = s.obstacle else { unreachable!() };
    let nx = w / db;
    let pb = |x: usize| (py / db) * nx + x / db;
    let probes = [pb(px + (2.0 * dlen) as usize), pb(w * 7 / 10)];
    let mut field = Field::new(w / db, h / db, db, db);
    let mut series: Vec<(f64, f64, f64)> = vec![];
    let mut stats = String::from("step,mean_ux,extent_D,max_extent_H,band_over_vortex\n");
    let mut orient: Vec<(usize, f64, f64)> = vec![];
    let t0 = Instant::now();
    let mut done = 0;
    while done < steps {
        for _ in 0..10 {
            match &mut e {
                E::S(g) => {
                    g.run(5);
                    g.accumulate_sums(db, db)
                }
                E::T(g) => {
                    g.run(5);
                    g.accumulate_sums(db, db)
                }
            }
        }
        done += 50;
        let sums: BlockSums = match &e {
            E::S(g) => pollster::block_on(g.take_sums()),
            E::T(g) => pollster::block_on(g.take_sums()),
        };
        field.add(&sums, 0.5);
        let v = |b: usize| sums.jy[b] as f64 * 0.5 * 3f64.sqrt() / sums.mass[b].max(1) as f64;
        series.push((done as f64, v(probes[0]), v(probes[1])));
        if done % stat_every == 0 {
            let (ext, emax, ratio) = wake_stats(&field.vorticity(), field.nx, field.ny, db, dlen, 0.25 * vmax);
            stats += &format!("{done},{:.5},{ext:.3},{emax:.3},{ratio:.4}\n", field.mean_ux());
            eprintln!("step {done} mean ux {:.4} extent {ext:.2} D max {emax:.2} H band/vortex {ratio:.3} ({:.0} s)", field.mean_ux(), t0.elapsed().as_secs_f64());
            if !field.mean_ux().is_finite() {
                println!("diverged at step {done}");
                break;
            }
        }
        if pngs.contains(&done) {
            write_png(&format!("{out}/vorticity-{done:06}.png"), &field.vorticity_rgba(vmax), field.nx, field.ny, 3);
            let (tilt, aspect, angle) = wake_orientation(&field.vorticity(), field.nx, field.ny, db, dlen);
            eprintln!("step {done} tilt {tilt:+.3} aspect {aspect:.2} angle {angle:.1}");
            orient.push((done, tilt, angle));
            let vb: Vec<u8> = field.vorticity().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out}/vort-{done:06}-{}x{}.f32", field.nx, field.ny), vb).unwrap();
        }
    }
    std::fs::write(format!("{out}/stats.csv"), &stats).unwrap();
    let mut probe = String::from("step,uy_2D,uy_07W\n");
    for r in &series {
        probe += &format!("{},{:.5},{:.5}\n", r.0, r.1, r.2);
    }
    std::fs::write(format!("{out}/probe.csv"), probe).unwrap();
    let half: Vec<&(f64, f64, f64)> = series.iter().filter(|r| r.0 > done as f64 / 2.0).collect();
    for (name, col) in [("probe 2D", 1), ("probe 0.7W", 2)] {
        let rows: Vec<(f64, f64)> = half.iter().map(|r| (r.0, if col == 1 { r.1 } else { r.2 })).collect();
        let m = rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64;
        let rms = (rows.iter().map(|r| (r.1 - m).powi(2)).sum::<f64>() / rows.len() as f64).sqrt();
        let (f, periods) = peak_frequency(&rows);
        println!("{name}: rms u_y {rms:.4}, peak frequency {f:.3e} per step ({periods:.1} periods), St = f D / (g U) = {:.3}", f * dlen / (g * u));
    }
    let late_o: Vec<&(usize, f64, f64)> = orient.iter().filter(|r| r.0 >= done / 2).collect();
    println!(
        "second half orientation: tilt {:+.3} (mean), angles {}",
        late_o.iter().map(|r| r.1).sum::<f64>() / late_o.len().max(1) as f64,
        late_o.iter().map(|r| format!("{:.0}", if r.2 > 135.0 { r.2 - 180.0 } else { r.2 })).collect::<Vec<_>>().join(", ")
    );
    let late: Vec<Vec<f64>> = stats.lines().skip(1).map(|l| l.split(',').map(|x| x.parse().unwrap()).collect()).filter(|v: &Vec<f64>| v[0] > done as f64 / 2.0).collect();
    let mean = |c: usize| late.iter().map(|v| v[c]).sum::<f64>() / late.len().max(1) as f64;
    println!(
        "second half: mean u_x {:.4}, extent {:.2} D, max extent {:.2} H, band/vortex {:.3}; {done} steps in {:.0} s",
        mean(1),
        mean(2),
        mean(3),
        mean(4),
        t0.elapsed().as_secs_f64()
    );
}

/// Orientation of the wake structures (`measure::orientation`, as the band
/// diagnosis's tilt.py): sigma D/4, columns [0.55, 0.85) of the width.
fn wake_orientation(o: &[f32], nx: usize, ny: usize, db: usize, dlen: f64) -> (f64, f64, f64) {
    measure::orientation(o, nx, ny, dlen / 4.0 / db as f64, nx * 55 / 100, nx * 85 / 100)
}

/// Wake indicators of raw vorticity dumps (`vort-<step>-<nx>x<ny>.f32`):
/// `bandstat --block 8 --plate 64 --u 0.2 file...`.
fn bandstat(args: &[String]) {
    let db = arg(args, "--block", 8usize);
    let rows = arg(args, "--plate", 64usize);
    let u = arg(args, "--u", 0.2f64);
    let dlen = rows as f64 * 3f64.sqrt() / 2.0;
    let vmax = (4.0 * u / dlen) as f32;
    println!("file,extent_D,max_extent_H,band_over_vortex,tilt,aspect,angle");
    for f in args.iter().filter(|a| a.ends_with(".f32")) {
        let dims = f.rsplit('-').next().unwrap().trim_end_matches(".f32");
        let (nx, ny) = dims.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap();
        let o: Vec<f32> = std::fs::read(f).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let (ext, emax, ratio) = wake_stats(&o, nx, ny, db, dlen, 0.25 * vmax);
        let (tilt, aspect, angle) = wake_orientation(&o, nx, ny, db, dlen);
        println!("{f},{ext:.3},{emax:.3},{ratio:.4},{tilt:+.3},{aspect:.2},{angle:.1}");
    }
}

/// Mirror check of the reference and the streaming mismatch: block fields
/// N and their y-mirror N' (blocks by -> nby-1-by, jy -> -jy); the lattice
/// automorphism G (x, y) -> (x + (y & 1), h-1-y) maps each site to its
/// mirror image (a horizontal mirror between rows composed with a half-site
/// shift in x) and direction d to M d (1 <-> 5, 2 <-> 4). Prints, per field
/// and direction, the mean |r| of the mismatch r_d = phi_d(source) -
/// phi_d(site), and the mean and max |r_d(x, y) - r'_Md(G(x, y))| (units of
/// 2^-12 particles), and the same for phi itself.
fn mirror_check(args: &[String]) {
    let b = arg(args, "--block", 16usize);
    let (w, h) = (arg(args, "--width", 256usize), arg(args, "--height", 128usize));
    let (nbx, nby) = (w / b, h / b);
    let rho = 28.0;
    let tau = 2.0 * std::f64::consts::PI;
    type BlockField = Box<dyn Fn(usize, usize) -> (f64, f64)>;
    let fields: Vec<(&str, BlockField)> = vec![
        ("shear ux(y) = 0.2 + 0.05 sin(2 pi y / H)", Box::new(move |_, by| (0.2 + 0.05 * (tau * (by as f64 + 0.5) / nby as f64).sin(), 0.0))),
        ("uy(y) = 0.05 sin(2 pi y / H) on ux = 0.2 (floor asymmetry in jy alone)", Box::new(move |_, by| (0.2, 0.05 * (tau * (by as f64 + 0.5) / nby as f64).sin()))),
        ("uy(x) = 0.05 sin(2 pi x / W) on ux = 0.2", Box::new(move |bx, _| (0.2, 0.05 * (tau * (bx as f64 + 0.5) / nbx as f64).sin()))),
        ("ux(x) = 0.2 + 0.05 sin(2 pi x / W)", Box::new(move |bx, _| (0.2 + 0.05 * (tau * (bx as f64 + 0.5) / nbx as f64).sin(), 0.0))),
        ("vortex-like: ux 0.2 + 0.03 sin(x) cos(y), uy -0.03 cos(x) sin(y)", Box::new(move |bx, by| {
            let (a, c) = (tau * (bx as f64 + 0.5) / nbx as f64 * 4.0, tau * (by as f64 + 0.5) / nby as f64 * 2.0);
            (0.2 + 0.03 * a.sin() * c.cos(), -0.03 * a.cos() * c.sin())
        })),
    ];
    let md = [0usize, 5, 4, 3, 2, 1, 6];
    // The collision tables under the y-mirror M: rule optimal's 12 images
    // of M s must be M of the images of s (as multisets), and rule
    // chiral's two chiralities must swap.
    let mir = |s: u16| -> u16 {
        let d = ternary::counts(s);
        (0..7).map(|c| d[md[c]] as u16 * ternary::POW3[c]).sum()
    };
    let t = ternary::Table::new();
    let (mut bad_opt, mut bad_chi) = (0, 0);
    for s0 in 0..ternary::NSTATE as u16 {
        let mut a: Vec<u16> = t.opt[mir(s0) as usize].to_vec();
        let mut b: Vec<u16> = t.opt[s0 as usize].iter().map(|&o| mir(o)).collect();
        a.sort();
        b.sort();
        bad_opt += (a != b) as u32;
        let c = t.chiral[mir(s0) as usize];
        bad_chi += (c[0] != mir(t.chiral[s0 as usize][1]) || c[1] != mir(t.chiral[s0 as usize][0])) as u32;
    }
    println!("collision tables under the y-mirror: rule optimal, {bad_opt} of {} states not equivariant; rule chiral, {bad_chi} not swapped", ternary::NSTATE);
    let p = Params { seed: 1, rule: Rule::Optimal, block: b, every: 1, n0: 4.0, boundary: None };
    for (name, f) in &fields {
        let mk = |mirror: bool| {
            let mut l = Signed::new(w, h, vec![signed::ZERO; w * h], p);
            for by in 0..nby {
                for bx in 0..nbx {
                    let sy = if mirror { nby - 1 - by } else { by };
                    let (ux, uy) = f(bx, sy);
                    l.nodes[by * nbx + bx] = signed::fixed_values(rho, ux, if mirror { -uy } else { uy });
                }
            }
            l.interpolated()
        };
        let (a, m) = (mk(false), mk(true));
        let g = |x: usize, y: usize| ((x + (y & 1)) % w, h - 1 - y);
        println!("{name} (B {b}, {w}x{h})");
        println!("dir,mean_abs_r,mean_diff_r,max_abs_diff_r,mean_abs_diff_phi");
        for d in 0..7 {
            let (mut sr, mut sd, mut md_max, mut sp, mut n) = (0.0f64, 0.0f64, 0i64, 0.0f64, 0.0f64);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let r = if d < 6 {
                        let (sx, sy) = fhp::source(x, y, d, w, h);
                        a[sy * w + sx][d] as i64 - a[i][d] as i64
                    } else {
                        0
                    };
                    let (gx, gy) = g(x, y);
                    let j = gy * w + gx;
                    let e = md[d];
                    let rm = if d < 6 {
                        let (sx, sy) = fhp::source(gx, gy, e, w, h);
                        m[sy * w + sx][e] as i64 - m[j][e] as i64
                    } else {
                        0
                    };
                    sr += r.abs() as f64;
                    sd += (r - rm) as f64;
                    md_max = md_max.max((r - rm).abs());
                    sp += (a[i][d] as i64 - m[j][e] as i64).abs() as f64;
                    n += 1.0;
                }
            }
            println!("{d},{:.2},{:+.3},{md_max},{:.2}", sr / n, sd / n, sp / n);
        }
    }
}

/// A single Gaussian vortex (core radius --radius, peak swirl --amp, sign
/// --sign) advected by a uniform flow --u on a periodic lattice, GPU
/// engine: an achiral scheme keeps it round whatever its sign. Raw
/// vorticity (display block --display) and PNGs at --png steps.
fn vortex(args: &[String]) {
    use fhp::coarse::Field;
    let w = arg(args, "--width", 1024usize);
    let h = arg(args, "--height", 512usize);
    let u = arg(args, "--u", 0.2f64);
    let r0 = arg(args, "--radius", 16.0f64);
    let amp = arg(args, "--amp", 0.05f64);
    let sign = arg(args, "--sign", 1.0f64);
    let db = arg(args, "--display", 4usize);
    let out = arg(args, "--out", "out".to_string());
    let pngs: Vec<usize> = list(args, "--png", "0,1000,2000,4000");
    std::fs::create_dir_all(&out).unwrap();
    let (_, p) = configs(args).remove(0);
    let rho = 7.0 * p.n0;
    let (xc, yc) = (w as f64 / 4.0, h as f64 * 3f64.sqrt() / 4.0);
    // u_theta = amp * 1.5852 (r0 / r) (1 - exp(-r^2 / r0^2)), peak amp at r = 1.12 r0
    let field = move |px: f64, py: f64| {
        let (dx, dy) = (px - xc, py - yc);
        let r2 = dx * dx + dy * dy;
        let g = if r2 < 1e-9 { 0.0 } else { sign * amp * 1.5852 * r0 * (1.0 - (-r2 / (r0 * r0)).exp()) / r2 };
        (rho, u - g * dy, g * dx)
    };
    let (cells, nodes) = signed::initial(w, h, &p, |_, _| false, field, p.seed);
    let nb = nodes.len();
    let mut g = pollster::block_on(signed::gpu::Gpu::new(w, h, p, || (cells, nodes, vec![[0; ternary::NCH]; nb]))).unwrap();
    println!("vortex B {} N {} sign {sign} r0 {r0} amp {amp} U {u} {w}x{h} columns {}", p.block, p.every, signed::COLUMNS.load(std::sync::atomic::Ordering::Relaxed));
    let last = *pngs.iter().max().unwrap();
    let mut done = 0;
    loop {
        if pngs.contains(&done) {
            let mut field = Field::new(w / db, h / db, db, db);
            for _ in 0..10 {
                g.run(5);
                g.accumulate_sums(db, db);
            }
            done += 50;
            field.add(&pollster::block_on(g.take_sums()), 1.0);
            let vmax = (amp * 4.0 / r0) as f32;
            write_png(&format!("{out}/vorticity-{:06}.png", done - 50), &field.vorticity_rgba(vmax), field.nx, field.ny, 2);
            let vb: Vec<u8> = field.vorticity().iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out}/vort-{:06}-{}x{}.f32", done - 50, field.nx, field.ny), vb).unwrap();
        }
        if done >= last {
            break;
        }
        let next = pngs.iter().copied().filter(|&s| s >= done).min().unwrap();
        g.run(next - done);
        done = next;
    }
    g.finish();
}

/// Whole-step mirror check (`measure::mirror_pairs`): a flow and its
/// y-mirror image over seeds; per statistic the mean half sum (the scheme's
/// bias, 0 if mirror-symmetric) with its standard error, and the mean half
/// difference (the flow's own handedness).
fn handed(args: &[String]) {
    let w = arg(args, "--width", 256usize);
    let h = arg(args, "--height", 128usize);
    let b = arg(args, "--block", 16usize);
    let every = arg(args, "--every", 4u32);
    let u = arg(args, "--u", 0.2f64);
    let a = arg(args, "--a", 0.1f64);
    let e = arg(args, "--e", 0.05f64);
    let steps = arg(args, "--steps", 2000usize);
    let ns = arg(args, "--seeds", 8u32);
    let db = arg(args, "--display", 8usize);
    let p = Params { seed: 1, rule: Rule::Optimal, block: b, every, n0: 4.0, boundary: None };
    let seeds: Vec<u32> = (1..=ns).collect();
    let t0 = Instant::now();
    let r = measure::mirror_pairs(p, w, h, u, a, e, &seeds, steps, db);
    println!("handed {w}x{h} B {b} N {every} U {u} a {a} e {e} steps {steps} seeds {ns} display {db} columns {}", signed::COLUMNS.load(std::sync::atomic::Ordering::Relaxed));
    for (s, (x, y)) in seeds.iter().zip(&r) {
        println!("seed {s}: flow {:+.4} {:+.4} {:+.4}  mirror {:+.4} {:+.4} {:+.4}", x[0], x[1], x[2], y[0], y[1], y[2]);
    }
    for (k, name) in ["u_x u_y corr", "vorticity tilt", "vorticity skew"].iter().enumerate() {
        let half: Vec<f64> = r.iter().map(|(x, y)| (x[k] + y[k]) / 2.0).collect();
        let n = half.len() as f64;
        let m = half.iter().sum::<f64>() / n;
        let se = (half.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        let d = r.iter().map(|(x, y)| (x[k] - y[k]) / 2.0).sum::<f64>() / n;
        println!("{name}: bias {m:+.4} +- {se:.4} ({:+.1} SE), flow {d:+.4}", m / se);
    }
    println!("{:.0} s", t0.elapsed().as_secs_f64());
}
