//! The dev page's constructors (`page::Page`, what `web::Sim` wraps) for
//! each model and block size, the GPU and CPU pages frame by frame, and
//! the ternary GPU path against the per-site reference with the page's
//! open boundary and plate.

use fhp::page::{ternary_cells, ternary_params, Model, Page};
use fhp::ternary::engine::Ternary;
use fhp::{street_for, SizeLimits, BLOCKS};

const MODELS: [Model; 3] = [Model::Fhp3, Model::Ternary, Model::Signed];

fn check_page(p: &mut Page, w: usize, h: usize, block: usize) {
    let s = p.setup;
    assert!(s.w >= w && s.h >= h && s.w.is_multiple_of(64) && s.h.is_multiple_of(16) && s.h.is_multiple_of(block), "{w}x{h} block {block} -> {}x{}", s.w, s.h);
    assert_eq!((p.field.nx, p.field.ny), (s.w / block, s.h / block));
    p.advance(12);
    let sums = pollster::block_on(p.sample()).expect("block sums");
    assert_eq!(sums.mass.len(), (s.w / block) * (s.h / block));
    assert_eq!(p.frame().len(), (s.w / block) * (s.h / block) * 4);
    assert_eq!(p.step(), 12);
    assert!(p.reynolds().is_finite());
}

#[test]
fn cpu_constructors_round_to_the_block_and_do_not_panic() {
    for model in MODELS {
        for block in BLOCKS {
            for (w, h) in [(1000, 300), (65, 3), (300, 47)] {
                let mut p = Page::cpu(model, w, h, 0.2, "randomize", block).unwrap_or_else(|e| panic!("{model:?} {w}x{h} block {block}: {e}"));
                check_page(&mut p, w, h, block);
            }
        }
        assert!(Page::cpu(model, 512, 64, 0.2, "randomize", 12).is_err());
        assert!(Page::cpu(model, 512, 64, 0.0, "randomize", 16).is_err());
        let e = Page::cpu(model, 60000, 30000, 0.2, "randomize", 16).err().expect("too large");
        assert!(e.contains("too large"), "{e}");
    }
    assert!(Model::parse("bogus").is_err());
    assert!(Page::cpu(Model::Ternary, 512, 64, 0.2, "sponge", 16).is_err());
    // Ternary holds 28 bytes per site: 16384 x 8192 (3.5 GiB) does not fit 2 GiB.
    assert!(Page::cpu(Model::Ternary, 16384, 8192, 0.2, "randomize", 16).is_err());
    // Signed holds 56: 16384 x 4096 (3.5 GiB) does not fit either.
    assert!(Page::cpu(Model::Signed, 16384, 4096, 0.2, "randomize", 16).is_err());
    assert!(Page::cpu(Model::Signed, 512, 64, 0.2, "sponge", 16).is_err());
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_constructors_round_to_the_block_and_do_not_panic() {
    for model in MODELS {
        for block in BLOCKS {
            for (w, h) in [(1000, 300), (65, 3)] {
                let mut p = match pollster::block_on(Page::gpu(model, w, h, 0.2, "randomize", block)) {
                    Ok(p) => p,
                    Err(e) if e.contains("adapter") => {
                        eprintln!("no GPU adapter; skipped");
                        return;
                    }
                    Err(e) => panic!("{model:?} {w}x{h} block {block}: {e}"),
                };
                check_page(&mut p, w, h, block);
            }
        }
        assert!(pollster::block_on(Page::gpu(model, 512, 64, 0.2, "randomize", 12)).is_err());
        let e = pollster::block_on(Page::gpu(model, 1 << 20, 1 << 17, 0.2, "randomize", 16)).err().expect("too large");
        assert!(e.contains("too large") || e.contains("adapter"), "{e}");
    }
}

/// The page's ternary GPU engine (table layout, open boundary, plate)
/// against the per-site reference, over several boundary refreshes, and
/// its block sums against the reference's for every block size.
#[cfg(feature = "gpu")]
#[test]
fn ternary_gpu_matches_the_reference_with_the_page_setup() {
    use fhp::ternary::gpu::{Gpu, Layout};
    let s = street_for(256, 64, 0.2, "randomize", 16, &SizeLimits::cpu(u64::MAX)).unwrap();
    let mut s = s;
    s.density = fhp::page::TERNARY_DENSITY;
    let (p, cells) = (ternary_params(&s), ternary_cells(&s));
    let mut r = Ternary::new(s.w, s.h, cells.clone(), p);
    let Ok(mut g) = pollster::block_on(Gpu::open(s.w, s.h, || cells.clone(), p.seed, Layout::Table, p.boundary)) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for n in [1usize, 31, 1, 40, 160] {
        r.run(n);
        g.run(n);
        assert_eq!(pollster::block_on(g.read_cells()), r.cells, "after {} steps", r.step);
        for b in BLOCKS {
            g.accumulate_sums(b, b);
            assert_eq!(pollster::block_on(g.take_sums()), r.block_sums(b, b), "block {b} after {} steps", r.step);
        }
    }
}

/// The page on the GPU and on the CPU gives the same block sums, frame by
/// frame, for every model (the engines are bit-identical).
#[cfg(feature = "gpu")]
#[test]
fn page_gpu_and_cpu_agree() {
    for model in MODELS {
        let mut c = Page::cpu(model, 384, 64, 0.2, "randomize", 16).unwrap();
        let Ok(mut g) = pollster::block_on(Page::gpu(model, 384, 64, 0.2, "randomize", 16)) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        for n in [10u32, 50, 77] {
            c.advance(n);
            g.advance(n);
            assert_eq!(pollster::block_on(g.sample()), pollster::block_on(c.sample()), "{model:?} step {}", c.step());
        }
        assert_eq!(g.frame(), c.frame());
    }
}

#[test]
fn signed_reference_block_follows_the_height() {
    use fhp::page::{signed_block, signed_every};
    assert_eq!((signed_block(2048, 512), signed_every(8)), (8, 1));
    assert_eq!((signed_block(4096, 1024), signed_every(16)), (16, 4));
    assert_eq!(signed_block(8192, 2048), 32);
    assert_eq!(signed_block(384, 64), 8);
    assert_eq!(Page::cpu(Model::Signed, 1024, 512, 0.2, "randomize", 16).unwrap().reference_block(), 8);
    assert_eq!(Page::cpu_ref(Model::Signed, 1024, 512, 0.2, "randomize", 16, 16).unwrap().reference_block(), 16);
    assert!(Page::cpu_ref(Model::Signed, 1024, 512, 0.2, "randomize", 16, 12).is_err());
    assert_eq!(Page::cpu_ref(Model::Ternary, 1024, 512, 0.2, "randomize", 16, 16).unwrap().reference_block(), 0);
}
