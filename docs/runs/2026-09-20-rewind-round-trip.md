# Rewind: reverse checkpoints and round-trip (C5 + C6)

Date: 2026-09-20 00:44–01:05. Reverse training on the desktop GPU (unit `jacobi-reverse`, log `logs/jacobi-reverse.log`), round-trip scoring on the development machine (wgpu).

## C5 — reverse checkpoints

Three runs, model20's recipe with `--init artifacts/model19 --reverse true --steps 250`, projections on
(`--project true --project-energy true --r-max 1.0001`), `--nu` ∈ {0, 0.001, 0.01}. Wall: ≈2 min each on the desktop GPU (0.48 s/step).

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

## Rewind bug: root cause and fix (2026-09-20)

Bisected by dumping `peak_location()` at each stage (drop, forward-20, each rewind tick) in and out of Playwright. The forward-20 state itself was already wrong ([0,10] instead of near [20,40]) whenever the smoke test's earlier RACE section ran first — narrowing it to state leaking across test sections, not the reverse model or the reverse step's math.

Root cause (`crates/web/src/lib.rs`, `drop_at`/`reset`): `PENDING_DROPS`/`PENDING_RESET` queued while `STEPPING` (from `drop_at`/`reset` calls that arrive mid-`step`) are drained only by `apply_pending`, which is called at the *start* of the next `step`/`rewind` call. But `drop_at`/`reset`, when called with `STEPPING` false, applied themselves immediately without first calling `apply_pending`. The smoke test's RACE section queues three drops at (10,10) during `step(30)`; they survive past the section's end (never drained, since the section's `step(30)` already finished by the time they're queued... no further step/rewind call follows inside RACE) and sit in `PENDING_DROPS` until the REWIND section's `step(20)` call flushes them — silently adding three extra drops on top of the (20,40) state right before the 20-tick rollout that `rewind()` later has to invert.

Fix: `drop_at` and `reset` now call `apply_pending(app)` before applying their own action when not `STEPPING`, so no queued item can leak into an unrelated later step/rewind call.

Rebuilt `web/pkg` (`wasm-pack build crates/web --target web --out-dir ../../web/pkg --release`, no `CARGO_TARGET_DIR` override). Headless result: **REWIND: PASS (peak at [20,40], origin [20,40], error 0.00 cells)**. `web/test/gallery.mjs` still passes (all 6 checkpoints load, ACCEPTANCE and COMPARE gates PASS).

The "rewind 20" button is now live on the demo page (`web/index.html` `#b-rewind20`, next to reset): it rewinds 20 ticks and reports `rewind: 20 ticks back, peak error N cells` in a one-line STATUS row, `N` measured against the peak location captured at the last tick-0 (drop/reset).
