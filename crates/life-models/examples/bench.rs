//! Time per generation of each model on the CPU (one thread, all threads)
//! and on WebGPU, at 16x16 and 2048x2048, with the fixture weights.
//!
//!     cargo run --release -p life-models --features wgpu,threads --example bench [bert|mlp2|stencil|-] [gpu]
//!
//! CPU: the whole `Model::step` (logits then threshold). WebGPU "step" is
//! upload + one generation + readback of the packed board; WebGPU "compute"
//! is the marginal cost of one generation inside a chain, (t(N gens) - t(1
//! gen)) / (N - 1), with the upload and readback paid once.

use std::path::PathBuf;
use std::time::Instant;

use life_models::{Kind, Model};

fn model(kind: Kind) -> Model {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(kind.id());
    Model::from_bytes(&std::fs::read(dir.join("model.safetensors")).unwrap(), &std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn board(w: usize, h: usize) -> Vec<u8> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..w * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 10 < 3) as u8
        })
        .collect()
}

struct Stat(Vec<f64>);

impl Stat {
    fn of(mut v: Vec<f64>) -> Stat {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Stat(v)
    }
    fn show(&self) -> String {
        let n = self.0.len();
        if n == 0 {
            return "-".into();
        }
        format!("{:.3} ({:.3}-{:.3})", self.0[n / 2], self.0[0], self.0[n - 1])
    }
}

fn ms<R>(f: impl FnOnce() -> R) -> f64 {
    let t = Instant::now();
    let r = f();
    std::hint::black_box(r);
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    println!("threads available: {}", std::thread::available_parallelism().unwrap());
    println!("| model | board | cpu 1 thread ms | cpu all threads ms | webgpu step ms | webgpu compute ms/gen |\n|---|---|---|---|---|---|");
    let only = std::env::args().nth(1).filter(|a| a != "-");
    let gpu_only = std::env::args().nth(2).as_deref() == Some("gpu");
    for kind in [Kind::Bert, Kind::Mlp2, Kind::Stencil] {
        if only.as_deref().is_some_and(|o| o != kind.id()) {
            continue;
        }
        let m = model(kind);
        let g = pollster::block_on(m.to_gpu());
        for (w, h) in [(16usize, 16usize), (2048, 2048)] {
            let b = board(w, h);
            let small = w == 16;
            let reps_cpu = if gpu_only { 0 } else if small { 200 } else { 3 };
            let cpu1 = Stat::of((0..reps_cpu).map(|_| ms(|| one.install(|| m.step(&b, w, h).unwrap()))).collect());
            let cpun = Stat::of((0..if gpu_only { 0 } else if small { 200 } else { 5 }).map(|_| ms(|| m.step(&b, w, h).unwrap())).collect());
            let (step, comp) = match &g {
                Ok(g) => {
                    for _ in 0..3 {
                        pollster::block_on(g.step(&b, w, h)).unwrap();
                    }
                    let step = Stat::of((0..if small { 100 } else { 9 }).map(|_| ms(|| pollster::block_on(g.step(&b, w, h)).unwrap())).collect());
                    let n = if small { 100u32 } else { 6 };
                    let comp = Stat::of(
                        (0..7)
                            .map(|_| {
                                let t1 = ms(|| pollster::block_on(g.run(&b, w, h, 1)).unwrap());
                                let tn = ms(|| pollster::block_on(g.run(&b, w, h, n)).unwrap());
                                (tn - t1) / (n - 1) as f64
                            })
                            .collect(),
                    );
                    (step.show(), comp.show())
                }
                Err(e) => (format!("no device: {e}"), String::new()),
            };
            println!("| {} | {w}x{h} | {} | {} | {step} | {comp} |", kind.id(), cpu1.show(), cpun.show());
        }
    }
    if let Ok(g) = &g_label() {
        println!("adapter: {g}");
    }
}

fn g_label() -> Result<String, life_models::Error> {
    pollster::block_on(model(Kind::Mlp2).to_gpu()).map(|g| g.adapter().to_string())
}
