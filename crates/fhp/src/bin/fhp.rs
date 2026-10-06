//! fhp CLI: `gen` (derive the collision circuit), `bench`, `run` (PNGs).

#![allow(clippy::needless_range_loop)]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("gen") => gen(),
        Some("bench") => bench(&args[1..]),
        Some("run") => run(&args[1..]),
        _ => {
            eprintln!("usage: fhp gen | bench | run");
            std::process::exit(2);
        }
    }
}

/// Search every variable order in both forms, keep the smallest circuit,
/// write src/collide_gen.rs and src/shaders/collide_gen.wgsl.
fn gen() {
    use fhp::synth::{permutations, synthesize, NIN};
    use rayon::prelude::*;
    let table = fhp::rules::table();
    let t0 = Instant::now();
    let perms = permutations(NIN);
    let best = [false, true]
        .into_par_iter()
        .flat_map(|flip| perms.par_iter().map(move |p| (flip, p)))
        .map(|(flip, p)| {
            let order: [usize; NIN] = p.as_slice().try_into().unwrap();
            let c = synthesize(&table, order, flip);
            (c.gate_count(), flip, order)
        })
        .min()
        .unwrap();
    let c = synthesize(&table, best.2, best.1);
    for x in 0..512u64 {
        let ins: [u64; NIN] = std::array::from_fn(|k| if (x >> k) & 1 == 1 { u64::MAX } else { 0 });
        let o = c.eval(ins);
        let want = table[(x >> 8) as usize][(x & 255) as usize];
        for k in 0..7 {
            assert_eq!(o[k] & 1, ((want >> k) & 1) as u64);
        }
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    std::fs::write(format!("{dir}/collide_gen.rs"), c.to_rust()).unwrap();
    std::fs::write(format!("{dir}/shaders/collide_gen.wgsl"), c.to_wgsl()).unwrap();
    println!(
        "searched {} orders x 2 forms in {:.1} s: {} gates {:?}, order {:?}, {} form",
        perms.len(),
        t0.elapsed().as_secs_f64(),
        c.gate_count(),
        c.counts(),
        c.order,
        if c.flip { "flip" } else { "direct" }
    );
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Site-updates per second of each path on a plate setup.
fn bench(args: &[String]) {
    use fhp::{bitsliced::Bitsliced, reference::Reference, Setup};
    let w = arg(args, "--width", 2048usize);
    let h = arg(args, "--height", 1024usize);
    let steps = arg(args, "--steps", 200usize);
    let outlet = fhp::boundary::Outlet::parse(&arg(args, "--outlet", fhp::DEFAULT_OUTLET.name().to_string())).expect("--outlet sponge|randomize|delete");
    let s = if args.iter().any(|a| a == "--periodic") { Setup::street_periodic(w, h) } else { Setup::street_with(w, h, 0.2, outlet) };
    let cells = s.initial_state();
    let sites = (w * h) as f64;
    let report = |name: &str, n: usize, secs: f64| {
        println!("{name:<28} {w}x{h} {n:>6} steps {secs:>8.3} s  {:>10.1} M site-updates/s", sites * n as f64 / secs / 1e6);
    };
    if !args.iter().any(|a| a == "--no-reference") {
        let mut r = Reference::new(w, h, cells.clone(), s.params);
        let n = (steps / 10).max(5);
        r.step();
        let t = Instant::now();
        r.run(n);
        report("reference (1 thread)", n, t.elapsed().as_secs_f64());
    }
    let cpu_threads: Vec<usize> = if args.iter().any(|a| a == "--no-cpu") { vec![] } else { vec![1, rayon::current_num_threads()] };
    for threads in cpu_threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut b = Bitsliced::from_cells(w, h, &cells, s.params);
        pool.install(|| {
            b.run(5);
            let t = Instant::now();
            b.run(steps);
            report(&format!("bitsliced ({threads} threads)"), steps, t.elapsed().as_secs_f64());
        });
    }
    if !args.iter().any(|a| a == "--no-gpu") {
        let mut g = pollster::block_on(fhp::gpu::Gpu::from_setup(&s)).expect("GPU adapter");
        g.run(50);
        g.finish();
        let n = steps * 20;
        let t = Instant::now();
        g.run(n);
        g.finish();
        report(&format!("gpu {}", g.adapter), n, t.elapsed().as_secs_f64());
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

/// Flow past a plate on the GPU (or `--cpu`): time-averaged block velocity,
/// vorticity and speed PNGs at the `--png` steps.
fn run(args: &[String]) {
    use fhp::coarse::Field;
    use fhp::{Obstacle, Params, Setup};
    let w = arg(args, "--width", 4096usize);
    let h = arg(args, "--height", 1024usize);
    let periodic = args.iter().any(|a| a == "--periodic");
    let outlet = fhp::boundary::Outlet::parse(&arg(args, "--outlet", fhp::DEFAULT_OUTLET.name().to_string())).expect("--outlet sponge|randomize|delete");
    let mut s = if periodic { Setup::street_periodic(w, h) } else { Setup::street_with(w, h, arg(args, "--u", 0.2), outlet) };
    s.density = arg(args, "--density", s.density);
    if let Some(b) = s.params.boundary {
        s.params.boundary = Some(fhp::boundary::Boundary::new(s.density, s.ux, b.inlet, b.outlet, b.kind));
    }
    let rows = arg(args, "--plate", s.plate_rows());
    s.obstacle = Obstacle::Plate { x: w / 6, y: h / 2 + 1, rows, cols: 2 };
    s.params.seed = arg(args, "--seed", 1u32);
    s.params.force_threshold = Params::with_force_probability(0, arg(args, "--force", s.params.force_probability())).force_threshold;
    s.params.walls = args.iter().any(|a| a == "--walls");
    let steps = arg(args, "--steps", 40000usize);
    // Block sums are taken every `every` steps and summed on the device; every
    // `frame` steps the sum is folded into the running average with weight `alpha`.
    let every = arg(args, "--every", 5usize);
    let frame = arg(args, "--frame", 50usize);
    let alpha = arg(args, "--alpha", 0.5f32);
    let (bw, bh) = (arg(args, "--bw", 16usize), arg(args, "--bh", 16usize));
    let scale = arg(args, "--scale", 3usize);
    let out = arg(args, "--out", "out".to_string());
    let pngs: Vec<usize> = arg(args, "--png", "10000,20000,30000,40000".to_string()).split(',').filter_map(|x| x.parse().ok()).collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    // Vorticity colour scale: a vortex of the plate's size turning at the flow speed.
    let vmax = arg(args, "--vmax", (4.0 * s.ux / (rows as f64 * 3f64.sqrt() / 2.0)) as f32);
    std::fs::create_dir_all(&out).unwrap();
    let nu = s.viscosity();
    let g = s.g();
    let d_real = rows as f64 * 3f64.sqrt() / 2.0;
    println!(
        "{w}x{h} density {:.3} (nu {:.4}, g {:.4}, nu/g {:.4}), plate {rows} rows ({d_real:.1} across), {}, force p {:.3e}, walls {}, {} path",
        s.density,
        nu,
        g,
        nu / g,
        match s.params.boundary {
            Some(b) => format!("open x: inflow U {} (Mach {:.3}, Re {:.0}), inlet {} cols, {} outlet {} cols", s.ux, Setup::mach(s.ux), s.reynolds(s.ux), b.inlet * 32, b.kind.name(), b.outlet * 32),
            None => "periodic x".to_string(),
        },
        s.params.force_probability(),
        s.params.walls,
        if cpu { "bit-sliced CPU" } else { "GPU" }
    );
    let cells = s.initial_state();
    enum Eng {
        Cpu(Box<fhp::bitsliced::Bitsliced>),
        Gpu(Box<fhp::gpu::Gpu>),
    }
    let mut eng = if cpu {
        Eng::Cpu(Box::new(fhp::bitsliced::Bitsliced::from_cells(w, h, &cells, s.params)))
    } else {
        Eng::Gpu(Box::new(pollster::block_on(fhp::gpu::Gpu::new(w, h, &cells, s.params)).expect("GPU adapter")))
    };
    let mut field = Field::new(w / bw, h / bh, bw, bh);
    let t = Instant::now();
    let mut done = 0;
    let mut acc: Option<fhp::coarse::BlockSums> = None;
    while done < steps {
        let n = every.min(steps - done);
        match &mut eng {
            Eng::Cpu(b) => {
                b.run(n);
                let x = b.block_sums(bw, bh);
                match acc.as_mut() {
                    Some(a) => a.add(&x),
                    None => acc = Some(x),
                }
            }
            Eng::Gpu(g) => {
                g.run(n);
                g.accumulate_sums(bw, bh);
            }
        }
        done += n;
        if done % frame != 0 && done != steps {
            continue;
        }
        let sums = match &mut eng {
            Eng::Cpu(_) => acc.take().unwrap(),
            Eng::Gpu(g) => pollster::block_on(g.take_sums()),
        };
        field.add(&sums, alpha);
        if done % 2000 == 0 || done == steps {
            let u = field.mean_ux() as f64;
            println!("step {done:>6}  mean ux {u:.4}  Re = U D g / nu = {:.0}  ({:.1} s)", s.reynolds(u), t.elapsed().as_secs_f64());
        }
        if pngs.contains(&done) {
            let vp = format!("{out}/vorticity-{done:06}.png");
            write_png(&vp, &field.vorticity_rgba(vmax), field.nx, field.ny, scale);
            write_png(&format!("{out}/speed-{done:06}.png"), &field.speed_rgba(2.0 * s.ux as f32), field.nx, field.ny, scale);
            println!("wrote {vp}");
        }
    }
}
