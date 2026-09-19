# Rewind: reverse checkpoints and round-trip (C5 + C6)

Date: 2026-09-20 00:44–01:05. Reverse training on the 3080 (unit `jacobi-reverse`, log `logs/jacobi-reverse.log`), round-trip scoring on the M2 (wgpu).

## C5 — reverse checkpoints

Three runs, model20's recipe with `--init artifacts/model19 --reverse true --steps 250`, projections on
(`--project true --project-energy true --r-max 1.0001`), `--nu` ∈ {0, 0.001, 0.01}. Wall: ≈2 min each on the 3080 (0.48 s/step).

| checkpoint | ν | final loss (step 200) |
|---|---:|---:|
| artifacts/model-rev-nu0 | 0 | 8.9e-2 |
| artifacts/model-rev-nu0.001 | 0.001 | see log |
| artifacts/model-rev-nu0.01 | 0.01 | see log |

## C6 — round-trip: forward N ticks with model20, then reverse N ticks with the reverse model

`./target/release/jacobi round-trip --forward artifacts/model20 --reverse artifacts/model-rev-nu<ν> --horizons 10,25,50 --trajectories 24 --seed 0 --nu <ν> --project true`
(raw output: `2026-09-20-round-trip-raw.txt`). Scores are against the original state; the last column is the RMSE of the physical N-tick change (what a "do nothing" rewind would score).

| ν | horizon | corr | amplitude ratio | round-trip RMSE | N-tick change RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.9600 | 0.9439 | 0.166 | 0.491 |
| 0 | 25 | 0.8513 | 0.8397 | 0.327 | 0.834 |
| 0 | 50 | 0.6449 | 0.7291 | 0.582 | 0.878 |
| 0.001 | 10 | 0.9817 | 0.9031 | 0.102 | 0.491 |
| 0.001 | 25 | 0.9263 | 0.8417 | 0.192 | 0.834 |
| 0.001 | 50 | 0.8025 | 0.7710 | 0.340 | 0.878 |
| 0.01 | 10 | 0.9773 | 0.9017 | 0.118 | 0.490 |
| 0.01 | 25 | 0.8998 | 0.8490 | 0.237 | 0.833 |
| 0.01 | 50 | 0.7466 | 0.7802 | 0.424 | 0.877 |

## Verdict

- Rewinding works for short horizons: at 10 ticks the round-trip recovers the start to corr 0.98 with ~1/5 of the physical change left as error; at 25 ticks corr 0.93; by 50 ticks corr 0.80 and the error is ~40 % of the change.
- The reverse model trained at the forward model's viscosity (ν = 0.001) is the best at every horizon; ν = 0 is the worst (reversing the diffusive part is the ill-posed piece). Amplitude comes back 10–25 % low: the reverse step damps.
- This is the "reversibility floor" for this pair; no classical reverse solver reference was run (the solver is not time-reversible with viscosity).

Demo: `web/model/model20/model-reverse.bin` + `meta-reverse.json` = model-rev-nu0.001 (gitignored); `rewind()` enabled when present. Headless check: `web/test/smoke.mjs` (REWIND line).

## Headless rewind check (2026-09-20 01:10)

`web/test/smoke.mjs` with model-rev-nu0.001 staged as `web/model/model20/model-reverse.bin`: all other gates PASS; **REWIND: FAIL** (peak at [9,9], origin [20,40], error 32.9 cells). The CLI round-trip at 25 ticks is corr 0.93, so this is most likely a web wiring bug (stale wasm export, wrong state passed to the reverse step, or peak finder on an empty field), not the model. Open item.
