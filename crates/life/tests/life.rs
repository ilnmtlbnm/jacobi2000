//! Parity gates for the three Life models. Weights are the published Burn
//! checkpoints converted to the PyTorch `state_dict` layout; references are
//! the Burn oracle logits (all 512 neighbourhoods for `bert` and `mlp2`, a
//! 16x16 board for `stencil`) and PyTorch logits on boards of several
//! shapes. `tools/make_fixtures.py` builds `tests/fixtures/`.
//!
//! The GPU tests need a WebGPU adapter and are skipped, with a line saying
//! so, when there is none.

use life::{is_alive, Kind, Model};
use serde_json::Value;

const TOL: f32 = 1e-4;

fn fixture(kind: Kind) -> Model {
    let (bytes, cfg): (&[u8], &str) = match kind {
        Kind::Bert => (include_bytes!("fixtures/bert/model.safetensors"), include_str!("fixtures/bert/config.json")),
        Kind::Mlp2 => (include_bytes!("fixtures/mlp2/model.safetensors"), include_str!("fixtures/mlp2/config.json")),
        Kind::Stencil => (include_bytes!("fixtures/stencil/model.safetensors"), include_str!("fixtures/stencil/config.json")),
    };
    Model::from_bytes(bytes, cfg).unwrap()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("fixtures/reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn cells(s: &str) -> Vec<u8> {
    s.bytes().map(|b| b - b'0').collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

const KINDS: [Kind; 3] = [Kind::Bert, Kind::Mlp2, Kind::Stencil];

/// The 3x3 torus holding neighbourhood case `k` (bits 0..8 are NW, N, NE, W,
/// E, SW, S, SE; bit 8 the cell): its centre cell sees exactly that case.
fn case_board(k: usize) -> Vec<u8> {
    let mut b = vec![0u8; 9];
    for (bit, pos) in [0, 1, 2, 3, 5, 6, 7, 8, 4].into_iter().enumerate() {
        b[pos] = ((k >> bit) & 1) as u8;
    }
    b
}

fn life_rule(case: usize) -> bool {
    let n = (0..8).filter(|b| (case >> b) & 1 == 1).count();
    let me = (case >> 8) & 1 == 1;
    matches!((me, n), (true, 2) | (true, 3) | (false, 3))
}

fn life_step(c: &[u8], w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| {
            let nb = life::cpu::taps(i % w, i / w, w, h);
            let n = nb.iter().filter(|&&j| c[j] != 0).count();
            matches!((c[i] != 0, n), (true, 2) | (true, 3) | (false, 3)) as u8
        })
        .collect()
}

fn rand_board(w: usize, h: usize, seed: u64, density: f64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..w * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 11) as f64 / (1u64 << 53) as f64) < density) as u8
        })
        .collect()
}

fn exhaustive_logits(kind: Kind, m: &Model) -> Vec<f32> {
    let k = kind.outputs();
    let mut out = Vec::new();
    for case in 0..512 {
        let lg = m.logits(&case_board(case), 3, 3).unwrap();
        out.extend_from_slice(&lg[4 * k..5 * k]);
    }
    out
}

#[test]
fn parameter_counts() {
    assert_eq!(fixture(Kind::Bert).cfg.num_params(), 3490);
    assert_eq!(fixture(Kind::Mlp2).cfg.num_params(), 1442);
    assert_eq!(fixture(Kind::Stencil).cfg.num_params(), 3329);
}

#[test]
fn cpu_exhaustive_next_state_matches_the_life_rule() {
    for kind in [Kind::Bert, Kind::Mlp2] {
        let lg = exhaustive_logits(kind, &fixture(kind));
        let wrong = (0..512).filter(|&c| is_alive(kind, &lg[c * 2..c * 2 + 2]) != life_rule(c)).count();
        println!("{}: {} of 512 neighbourhoods differ from the Life rule", kind.id(), wrong);
        assert_eq!(wrong, 0, "{}", kind.id());
    }
}

#[test]
fn cpu_exhaustive_logits_match_references() {
    let r = reference();
    for kind in KINDS {
        let lg = exhaustive_logits(kind, &fixture(kind));
        let torch = floats(&r["torch"]["exhaustive"][kind.id()]);
        let d_torch = max_diff(&lg, &torch);
        println!("{}: cpu vs PyTorch, 512 neighbourhoods, max abs logit diff {:.3e}", kind.id(), d_torch);
        assert!(d_torch < TOL, "{}: {d_torch}", kind.id());
        if kind != Kind::Stencil {
            let burn = floats(&r["burn"][kind.id()]);
            let d_burn = max_diff(&lg, &burn);
            println!("{}: cpu vs Burn oracle, 512 neighbourhoods, max abs logit diff {:.3e}", kind.id(), d_burn);
            assert!(d_burn < TOL, "{}: {d_burn}", kind.id());
        }
    }
}

#[test]
fn cpu_stencil_matches_burn_on_its_16x16_board() {
    let r = reference();
    let b = &r["burn"]["stencil"];
    let (w, h) = (b["w"].as_u64().unwrap() as usize, b["h"].as_u64().unwrap() as usize);
    let lg = fixture(Kind::Stencil).logits(&cells(b["cells"].as_str().unwrap()), w, h).unwrap();
    let d = max_diff(&lg, &floats(&b["logits"]));
    println!("stencil: cpu vs Burn oracle, 16x16, max abs logit diff {d:.3e}");
    assert!(d < TOL);
}

#[test]
fn cpu_boards_match_pytorch() {
    let r = reference();
    for kind in KINDS {
        let m = fixture(kind);
        let mut worst = 0.0f32;
        for b in r["torch"]["boards"].as_array().unwrap() {
            let (w, h) = (b["w"].as_u64().unwrap() as usize, b["h"].as_u64().unwrap() as usize);
            let lg = m.logits(&cells(b["cells"].as_str().unwrap()), w, h).unwrap();
            let d = max_diff(&lg, &floats(&b[kind.id()]));
            assert!(d < TOL, "{} {w}x{h}: {d}", kind.id());
            worst = worst.max(d);
        }
        println!("{}: cpu vs PyTorch, {} boards incl. 37x23, max abs logit diff {:.3e}", kind.id(), r["torch"]["boards"].as_array().unwrap().len(), worst);
    }
}

#[test]
fn cpu_step_is_the_argmax_of_the_logits() {
    for kind in KINDS {
        let m = fixture(kind);
        let c = rand_board(37, 23, 5, 0.4);
        let lg = m.logits(&c, 37, 23).unwrap();
        let s = m.step(&c, 37, 23).unwrap();
        for (i, l) in lg.chunks(kind.outputs()).enumerate() {
            assert_eq!(s[i] == 1, is_alive(kind, l));
        }
    }
}

/// `bert` and `mlp2` are exact on every neighbourhood, so a board evolves
/// like Life itself, on any shape, wrap-around included.
#[test]
fn cpu_boards_evolve_like_life() {
    for kind in [Kind::Bert, Kind::Mlp2] {
        let m = fixture(kind);
        for (w, h) in [(37, 23), (16, 16), (2, 5), (1, 4)] {
            let mut c = rand_board(w, h, 7, 0.4);
            for _ in 0..5 {
                let want = life_step(&c, w, h);
                c = m.step(&c, w, h).unwrap();
                assert_eq!(c, want, "{} {w}x{h}", kind.id());
            }
        }
    }
}

#[test]
fn stencil_on_a_16x16_board_follows_life() {
    let m = fixture(Kind::Stencil);
    let mut c = rand_board(16, 16, 3, 0.28);
    let mut wrong = 0;
    for _ in 0..5 {
        let want = life_step(&c, 16, 16);
        let got = m.step(&c, 16, 16).unwrap();
        wrong += got.iter().zip(&want).filter(|(a, b)| a != b).count();
        c = want;
    }
    println!("stencil: {wrong} wrong cells over 5 generations of 16x16 (teacher-forced)");
    assert_eq!(wrong, 0);
}

#[test]
fn bad_boards_and_configs_are_errors() {
    let m = fixture(Kind::Mlp2);
    assert!(m.logits(&[0; 5], 2, 3).is_err());
    assert!(m.logits(&[], 0, 0).is_err());
    assert!(Model::from_bytes(include_bytes!("fixtures/bert/model.safetensors"), r#"{"model":"mlp2","hidden":32}"#).is_err());
    assert!(Model::from_bytes(include_bytes!("fixtures/bert/model.safetensors"), r#"{"model":"bert","d_model":16,"n_layers":2}"#).is_err());
}

#[cfg(feature = "wgpu")]
mod gpu {
    use super::*;
    use life::gpu::GpuModel;

    fn gpu(kind: Kind) -> Option<(Model, GpuModel)> {
        let m = fixture(kind);
        match pollster::block_on(m.to_gpu()) {
            Ok(g) => Some((m, g)),
            Err(e) => {
                println!("skipped: no WebGPU device ({e})");
                None
            }
        }
    }

    #[test]
    fn exhaustive_matches_references_and_the_life_rule() {
        let r = reference();
        for kind in KINDS {
            let Some((_, g)) = gpu(kind) else { return };
            let k = kind.outputs();
            let (mut lg, mut next) = (Vec::new(), Vec::new());
            for case in 0..512 {
                let b = case_board(case);
                let l = pollster::block_on(g.logits(&b, 3, 3)).unwrap();
                lg.extend_from_slice(&l[4 * k..5 * k]);
                next.push(pollster::block_on(g.step(&b, 3, 3)).unwrap()[4]);
            }
            let d_torch = max_diff(&lg, &floats(&r["torch"]["exhaustive"][kind.id()]));
            println!("{}: gpu ({}) vs PyTorch, 512 neighbourhoods, max abs logit diff {:.3e}", kind.id(), g.adapter(), d_torch);
            assert!(d_torch < TOL, "{}: {d_torch}", kind.id());
            if kind != Kind::Stencil {
                let d_burn = max_diff(&lg, &floats(&r["burn"][kind.id()]));
                println!("{}: gpu vs Burn oracle, 512 neighbourhoods, max abs logit diff {:.3e}", kind.id(), d_burn);
                assert!(d_burn < TOL);
                let wrong = (0..512).filter(|&c| (next[c] == 1) != life_rule(c)).count();
                println!("{}: gpu step, {} of 512 neighbourhoods differ from the Life rule", kind.id(), wrong);
                assert_eq!(wrong, 0);
            }
            let want: Vec<u8> = (0..512).map(|c| is_alive(kind, &lg[c * k..(c + 1) * k]) as u8).collect();
            assert_eq!(next, want, "{}: gpu step disagrees with its own logits", kind.id());
        }
    }

    #[test]
    fn boards_match_pytorch_and_the_cpu() {
        let r = reference();
        for kind in KINDS {
            let Some((m, g)) = gpu(kind) else { return };
            let (mut worst_t, mut worst_c) = (0.0f32, 0.0f32);
            for b in r["torch"]["boards"].as_array().unwrap() {
                let (w, h) = (b["w"].as_u64().unwrap() as usize, b["h"].as_u64().unwrap() as usize);
                let c = cells(b["cells"].as_str().unwrap());
                let lg = pollster::block_on(g.logits(&c, w, h)).unwrap();
                let d = max_diff(&lg, &floats(&b[kind.id()]));
                assert!(d < TOL, "{} {w}x{h}: {d}", kind.id());
                worst_t = worst_t.max(d);
                worst_c = worst_c.max(max_diff(&lg, &m.logits(&c, w, h).unwrap()));
                assert_eq!(pollster::block_on(g.step(&c, w, h)).unwrap(), m.step(&c, w, h).unwrap(), "{} {w}x{h}", kind.id());
            }
            println!("{}: gpu vs PyTorch on the fixture boards incl. 37x23: {:.3e}; gpu vs cpu: {:.3e}", kind.id(), worst_t, worst_c);
        }
    }

    #[test]
    fn stencil_matches_burn_on_its_16x16_board() {
        let Some((_, g)) = gpu(Kind::Stencil) else { return };
        let r = reference();
        let b = &r["burn"]["stencil"];
        let lg = pollster::block_on(g.logits(&cells(b["cells"].as_str().unwrap()), 16, 16)).unwrap();
        let d = max_diff(&lg, &floats(&b["logits"]));
        println!("stencil: gpu vs Burn oracle, 16x16, max abs logit diff {d:.3e}");
        assert!(d < TOL);
    }

    /// Tile edges (8x8 workgroups), boards narrower than a tile and the
    /// non-square shapes, against the CPU path.
    #[test]
    fn odd_shapes_match_the_cpu() {
        for kind in KINDS {
            let Some((m, g)) = gpu(kind) else { return };
            let mut worst = 0.0f32;
            for (w, h) in [(1, 1), (1, 9), (2, 2), (3, 3), (7, 9), (8, 8), (9, 8), (8, 9), (15, 17), (16, 33), (37, 23), (130, 70), (301, 5)] {
                let c = rand_board(w, h, (w * 31 + h) as u64, 0.4);
                let lg = pollster::block_on(g.logits(&c, w, h)).unwrap();
                let d = max_diff(&lg, &m.logits(&c, w, h).unwrap());
                assert!(d < TOL, "{} {w}x{h}: {d}", kind.id());
                worst = worst.max(d);
            }
            println!("{}: gpu vs cpu logits on 13 shapes (1x1 to 301x5): {:.3e}", kind.id(), worst);
        }
    }

    #[test]
    fn generations_chain_without_readback() {
        for kind in KINDS {
            let Some((m, g)) = gpu(kind) else { return };
            let c0 = rand_board(37, 23, 11, 0.4);
            let mut c = c0.clone();
            let mut states = vec![];
            for _ in 0..6 {
                c = m.step(&c, 37, 23).unwrap();
                states.push(c.clone());
            }
            for gens in 1..=6u32 {
                let got = pollster::block_on(g.run(&c0, 37, 23, gens)).unwrap();
                assert_eq!(got, states[gens as usize - 1], "{} after {gens} generations", kind.id());
            }
            assert_eq!(pollster::block_on(g.run(&c0, 37, 23, 0)).unwrap(), c0);
        }
    }

    /// Where Burn's stencil gather broke (3840x2160 at 8.3M cells): CPU and
    /// GPU logits on the same non-square board. Slow on the CPU; run with
    /// `--ignored`.
    #[test]
    #[ignore]
    fn full_hd_and_4k_boards() {
        for kind in KINDS {
            let Some((m, g)) = gpu(kind) else { return };
            for (w, h) in [(1920, 1080), (3840, 2160)] {
                let c = rand_board(w, h, 9, 0.3);
                let lg = pollster::block_on(g.logits(&c, w, h)).unwrap();
                let want = m.logits(&c, w, h).unwrap();
                let d = max_diff(&lg, &want);
                println!("{}: gpu vs cpu logits at {w}x{h}: {d:.3e}", kind.id());
                assert!(d < TOL);
            }
        }
    }
}
