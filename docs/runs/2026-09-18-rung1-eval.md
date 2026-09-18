# Rung-1 re-score of run 2026-09-18-8

Machine: Apple M2, 16 GB. No timings reported (metrics only). GPU checked
uncontended before the run: `ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'`
printed nothing.

- commit: `913ad8f` (the eval binary: `crates/train/src/metrics.rs` + the
  `eval-metrics` subcommand)
- backend: wgpu (Metal)
- checkpoint: `artifacts/model10` — run 8's log does not name its `--out`
  directory; this one reproduces run 8's rollout table exactly
  (`eval --model artifacts/model10 --trajectories 4 --seed 1000 --project false`
  gives 0.13504 / 0.87667 / 4.01738 at 10 / 50 / 200 ticks against the logged
  0.13504 / 0.87667 / 4.01736), and no other checkpoint in `artifacts/` does.
- command:
  `cargo run --release -p cli --features wgpu --no-default-features -- eval-metrics --model artifacts/model10 --horizons 10,25,50 --trajectories 24 --seed 0 --project false`
- `--project false` matches run 8's training command.
- trajectories: 24 scored, 22 skipped. Each is a fresh single drop (random
  position, amplitude 0.03-0.07, sigma 3.0-5.5, no velocity field); a
  trajectory is skipped when its first-reflection horizon
  (distance to the nearest wall / `sqrt(g·depth)·dt/dx`, 0.283 cells/tick
  here) is under 50 ticks, so nothing below is scored after the wave first
  touches a wall.
- bands are Laplacian-pyramid RMSE of the h error, wavelengths in cells.
- correlation, active-region RMSE, band and global RMSE are in normalized
  field units; amplitude and radius are on the decoded anomaly `h - depth`.

## Rung-1 metrics (mean over 24 trajectories)

| field | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err (cells) | band 2-4 | band 4-8 | band 8-16 | global RMSE (previously gameable) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | 10 | 0.9815 | 0.9875 | 0.27312 | 1.2159 | +0.015 | 0.04339 | 0.04057 | 0.04506 | 0.09191 |
| frozen field | 10 | 0.7134 | 0.9687 | 1.14678 | 1.6385 | -0.012 | 0.04400 | 0.08890 | 0.12711 | 0.35631 |
| model | 25 | 0.7409 | 0.7099 | 0.76760 | 0.8947 | -0.260 | 0.09205 | 0.10845 | 0.16461 | 0.36725 |
| frozen field | 25 | 0.1099 | 0.1387 | 1.38752 | 3.6404 | -5.195 | 0.13832 | 0.28877 | 0.49053 | 0.65980 |
| model | 50 | 0.2642 | 0.3889 | 0.82966 | 1.3417 | -10.626 | 0.10690 | 0.15660 | 0.24742 | 0.57123 |
| frozen field | 50 | -0.2064 | -0.4485 | 1.23478 | 3.7638 | -10.787 | 0.13034 | 0.27024 | 0.53766 | 0.76121 |

## Acceptance bar (GOAL-2026-09-18)

| quantity | bar | run 8 |
|---|---|---:|
| correlation at 25 ticks | >= 0.9 | 0.7409 |
| correlation at 50 ticks | >= 0.8 | 0.2642 |
| amplitude ratio at 50 ticks | 0.7-1.3 | 1.3417 |
| wavefront radius error at 50 ticks | < 2 cells | 10.626 |
