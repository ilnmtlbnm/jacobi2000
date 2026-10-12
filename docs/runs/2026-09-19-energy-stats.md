# Energy stats — 2026-09-19 (task A2, measuring the bound from data)

Machine: a 16 GB consumer machine. Backend: ndarray (CPU) — `energy-stats` needs no GPU.

- commit: `6fac81c`
- command: `./target/release/jacobi energy-stats --pool 32 --ticks 100 --nu 0.001 --seed 42`
- generator: single-drop 70/30 mixture (`--extended-fraction 0.3`, `--long-fraction 0.2`,
  `--amplitude-max 0.3`, `--extra-drop-prob 0.3`, `--no-horizon-fraction 0.3`), same
  distribution as run 17/18's training pool, `nu = 1e-3`.

## Per-tick total-energy ratio `r_t = E_{t+1}/E_t`, 3200 samples (32 trajectories x 100 ticks)

| min | p50 | p99 | max |
|---:|---:|---:|---:|
| 0.999997 | 1.000000 | 1.000000 | 1.000002 |

## E in physical units, 3200 samples

| min | p50 | p99 | max |
|---:|---:|---:|---:|
| 19734.691406 | 20129.197266 | 20677.054688 | 20677.894531 |

Total energy is dominated by the flat pond's potential energy baseline
(`0.5 * g * depth^2` over 4096 cells ~ 20090), which the drop only perturbs by a
small fraction — this is why `r_t` sits essentially at 1 rather than tracking the
kinetic-energy dissipation rate directly: the solver is very close to energy-neutral
at this `nu`, apart from float32 rounding.

## Initial transient (fresh drop, ticks 1-5 from birth, 32 trajectories each)

| tick | min | p50 | p99 | max | n |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.995565 | 1.000000 | 1.000002 | 1.000002 | 32 |
| 2 | 0.999999 | 1.000000 | 1.000001 | 1.000001 | 32 |
| 3 | 0.999998 | 1.000000 | 1.000003 | 1.000003 | 32 |
| 4 | 0.999998 | 1.000000 | 1.000003 | 1.000003 | 32 |
| 5 | 0.999999 | 1.000000 | 1.000002 | 1.000002 | 32 |

Tick 1's min (0.995565) is one outlier trajectory at drop birth (the drop is added
directly to `h`, an initial condition the continuous solver never produced, so its
first step is the biggest one-off energy adjustment in any trajectory's life); ticks
2-5 and the all-tick distribution above agree the steady-state ratio is ~1 to 1e-6.

## Bound chosen

`r_max = measured max * (1 + 1e-4) = 1.000002 * 1.0001 ~ 1.000102`, i.e.
`--r-max 1.000102` for training and eval (task A2, `project_energy` in
`crates/jacobi2000/src/project.rs`).

For comparison, the previous coded bound (`1 - c*nu*dt`, `c = 1`, `nu = 1e-3`,
`dt ~ 0.0903`) gives `~0.999910` — tighter than every measured ratio above,
including the p99, which is why it stripped kinetic energy from the model's
predictions every step (docs/runs/2026-09-19-energy-smoke-2.md, loss ~0.4 instead
of ~1e-3). The formula is kept as `train::fit::fallback_r_max` for when no
measurement exists at a given `nu`, but is no longer the default.
