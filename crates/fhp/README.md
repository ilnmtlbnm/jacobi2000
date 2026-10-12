# fhp

The FHP lattice gas (Frisch, Hasslacher and Pomeau, 1986) with the FHP-III collision set, as one Rust crate that runs natively and in the browser. There are three engines, and all three are bit-identical:

- `reference.rs` stores one byte per site and looks the collision up in a 2 x 256 table.
- `bitsliced.rs` stores one bit-plane per u64 word (64 sites). The collision is the boolean circuit `collide_gen.rs`, derived from the table, and streaming is done with shifts. Rows run on a rayon pool when the `threads` feature is on.
- `gpu.rs` runs the same bit-planes as u32 words in WGSL (`shaders/step.wgsl` plus the generated `shaders/collide_gen.wgsl`) through wgpu: Metal natively, WebGPU in the browser.

## Model

- The hexagonal lattice is stored on a W x H square grid, with odd rows shifted half a site to the right (d'Humieres and Lallemand 1987, section 3.1). Each site has bits 0 to 5 for the moving directions E, NE, NW, W, SW and SE, bit 6 for the rest particle and bit 7 for the obstacle. Streaming is periodic in both directions; W must be a multiple of 64 and H must be even. The page and `street_for` round a requested size up to a multiple of 64 by a multiple of 16 and of the coarse-graining block (8, 16 or 32), and refuse sizes that do not fit the device, naming the largest one that does.
- Every engine runs the same step: `state <- boundary(force(collide(stream(state))))`. The stored state is always post-collision.
- The collisions are FHP-III, model III of d'Humieres and Lallemand (1987, section 2.2). States are grouped by mass and momentum, and a state in a group of q > 1 configurations goes to each of the others with probability 1/(q-1). The two zero-momentum groups of five are the exception: each splits into a group of three (a rest particle plus a head-on pair, or two head-on pairs) and a group of two (the symmetric triples), and the two parts do not exchange. Every group then has size 1, 2 or 3, so one random chirality bit per site gives the published probabilities exactly. A pair swaps, and a group of three steps one way round a rotation-equivariant 3-cycle for chirality 0 and the other way for chirality 1. 76 of the 128 configurations change, which is the paper's count.
- Obstacles use bounce-back: an obstacle site reverses every moving bit.
- Forcing: on each step, every fluid site that holds a west-mover and no east-mover flips it to an east-mover with probability p. This conserves mass and adds 2 to the x momentum per flip. Within each 32-site chunk at most one site is forced per step: the chunk's hash h forces site `h & 31` when `h >> 5 < p * 2^32`.
- Open boundaries in x (the default setup): every 32nd step, after the collision, the boundary chunks of every row are refreshed in one small extra pass (on the GPU its own dispatch, so the main step kernel is the periodic one). The inlet (the first 32 columns) is replaced with fresh equilibrium samples. The outlet is one of three kinds. `randomize`, the default, replaces the last 64 columns with fresh equilibrium samples, the same as the inlet. `sponge` replaces sites over the last W/16 columns with a probability that ramps quadratically from 0 to 1 across them. `delete` empties the last 64 columns, which makes a low-density sink (kept to show that artifact). The inlet is 32 columns wide, so what streams around from the outlet between refreshes (at most 31 columns) stays inside it. A replaced fluid site gets a fresh, independent sample of the Fermi-Dirac equilibrium N_i = 1/(1 + exp(h + a c_ix)) (d'Humieres and Lallemand 1987, eq. 2.2; the general derivation is in Frisch et al. 1987), with h and a solved by Newton's method so that the mass is 7d and the x momentum 7dU at the inflow density d and speed U. This is the same "wind-tunnel" idea as d'Humieres and Lallemand 1987, section 3.1: fresh particles drawn from the equilibrium at one end, an output flux at the other. Every Bernoulli draw compares 12 random words bit-serially against a 12-bit threshold, one random stream per bit, so all three engines get identical bits. The old fully periodic setup with forcing is `Setup::street_periodic` (`fhp run --periodic`).
- Free-slip walls (`walls`, `fhp run --walls`): rows 0 and H-1 mirror the moving bits about the x axis (NE <-> SE, NW <-> SW), with no collision and no forcing, so nothing crosses the periodic seam in y. Off by default; y is periodic.
- Random bits come from lowbias32 (Chris Wellons, hash-prospector) applied to (seed, step, stream, row, 32-site chunk). This makes them independent of word width and thread count, which is what lets the three engines agree bit for bit.
- The initial state samples the linearized Fermi-Dirac equilibrium (d'Humieres and Lallemand 1987, eq. 2.5) at density d per cell and velocity u.
- Default flow (`Setup::street`): density 0.28 per cell, inflow U = 0.2 (Mach 0.306 with c_s = sqrt(3/7)), and a plate H/8 rows tall at W/6. That gives Re 75 at 4096 x 1024, 150 at 8192 x 2048, and 300 at 16384 x 4096.
- Transport coefficients use d'Humieres and Lallemand 1987, eqs. 2.11 and 2.13: nu = 1/(28 d(1-d)(1 - 8 d(1-d)/7)) - 1/8 and g = (7/12)(1-2d)/(1-d). The Reynolds number is Re = U D g / nu, with D the plate length across the flow (rows times sqrt(3)/2).

## Collision circuit

`cargo run --release -p fhp -- gen` derives the circuit from `rules::table()`. For each of the 9! orders of the nine inputs (six moving bits, rest, obstacle, chirality), and for both the direct form and the flip form (output XOR input), it builds a reduced ordered BDD per output, shares nodes across the seven outputs, and lowers each node to at most three gates with structural hashing. It keeps the smallest result. The current circuit has 531 gates: 162 AND, 10 AND-NOT, 16 OR, 9 OR-NOT, 322 XOR and 12 NOT. It uses the direct form with variable order [obstacle, E, NE, W, SW, chirality, NW, SE, rest]. A test checks it exhaustively against the table, and another test regenerates it from the recorded order and compares the result with the committed Rust and WGSL.

## Run

```bash
cargo test --release -p fhp                      # the GPU test needs an adapter
cargo run --release -p fhp -- bench --width 4096 --height 1024 --steps 400
cargo run --release -p fhp -- run --out out      # flow past a plate (--u, --outlet, --walls, --periodic, --width, --height)
ENGINE_BUILD=<tag> scripts/build_fhp_web.sh      # pkg (WebGPU + single thread) and pkg-mt (threads)
python3 scripts/serve_coi.py --dir crates/fhp --port <port>   # COOP/COEP for wasm threads
# then open http://127.0.0.1:<port>/www/index.html (?path=webgpu|threads|single, ?w=&h=, ?u=, ?mode=randomize|sponge|delete|periodic,
#   ?model=fhp3|ternary|signed, ?block=8|16|32, ?side=1 for FHP-III, ternary and signed stacked)
PLAYWRIGHT=<.../playwright/index.mjs> node scripts/fhp_headless.mjs http://127.0.0.1:<port>/www/index.html
```

## Ternary FHP

`src/ternary/` is the same lattice with 0, 1 or 2 particles per channel (an integer lattice gas, Boghosian et al., PRE 55, 4137 (1997)): a per-site engine with four collision rules, a bit-sliced engine and two GPU engines for rule `chiral`, all bit-identical, and the measurements against FHP-III (`cargo run --release -p fhp --bin ternary -- theory|shear|advect|noise|street|bench|gen|summarize`).

## Signed ternary FHP

`src/signed/` stores each channel as a deviation in {-1, 0, +1} around a per-block reference equilibrium. The reference, the equilibrium, the rounding and the reservoirs are fixed point (i32 in units of 2^-12 particles), so the CPU engine and the GPU engine (`signed/gpu.rs`, `shaders/signed.wgsl`) are bit-identical and mass and momentum are conserved exactly in those units (`tests/signed.rs`, `tests/signed_gpu.rs`). The GPU engine has rules `optimal` and `chiral`, the open boundary, the plate and block sums.

## Sources

- U. Frisch, B. Hasslacher, Y. Pomeau, "Lattice-gas automata for the Navier-Stokes equation," Phys. Rev. Lett. 56, 1505 (1986). https://doi.org/10.1103/PhysRevLett.56.1505
- U. Frisch, D. d'Humieres, B. Hasslacher, P. Lallemand, Y. Pomeau, J.-P. Rivet, "Lattice gas hydrodynamics in two and three dimensions," Complex Systems 1, 649-707 (1987). https://content.wolfram.com/sites/13/2018/02/01-4-7.pdf (section 2.2 defines FHP-I, II and III, with FHP-III as the collision-saturated FHP-II)
- D. d'Humieres, P. Lallemand, "Numerical simulations of hydrodynamics with lattice gas automata in two dimensions," Complex Systems 1, 599-632 (1987). https://content.wolfram.com/sites/13/2018/02/01-4-5.pdf (model III rules and the count of 76 active configurations, eqs. 2.5, 2.11 and 2.13, the rectangular storage with row-parity propagation, bounce-back, and section 6.3 with its flow-past-plate setup: 2816 x 1024 lattice, d = 0.30, u = 0.428, model III, plate 216.5 across, Re about 300, a von Karman street after 40000 iterations in figure 10)
- D. d'Humieres, Y. Pomeau, P. Lallemand, "Simulation d'allees de von Karman bidimensionnelles a l'aide d'un gaz sur reseau," C. R. Acad. Sci. Paris II 301, 1391 (1985).
- daleobrien/lattice-gas, https://github.com/daleobrien/lattice-gas (a native Rust FHP code, 6144 x 1536 lattice with a plate 173 cells across, Re about 100, density 0.22 per direction, 20-cell blocks and a 25-step running average. Its collision draws uniformly from the whole mass-momentum group, and its inlet is re-seeded each step instead of forced.)
- L. P. Kadanoff, G. R. McNamara, G. Zanetti, "From automata to fluid flow: comparisons of simulation and theory," Phys. Rev. A 40, 4527 (1989): a forced FHP channel flow.
- C. Wellons, hash-prospector, lowbias32. https://github.com/skeeto/hash-prospector
