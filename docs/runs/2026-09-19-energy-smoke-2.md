# Energy smoke 2 — 2026-09-19 (task A1, after the detach fix)

Machine: Apple M2 16 GB. wgpu (Metal). Timings **provisional** (llm-life engine tests shared the GPU: ~5 s/step).
- commit: `666119c` (detach the energy-projection scale; gradient-finiteness test)
- command: `./target/release/jacobi train --init artifacts/model19 --mask stencil --radius 1 --dilation 1 --steps 300 --single-drop true --extended-fraction 0.3 --amplitude-max 0.3 --extra-drop-prob 0.3 --no-horizon-fraction 0.3 --project true --project-energy true --nu 0.001 --rollout 4 --rollout-after 0 --noise 0.04 --spectral 1 --lr 0.0001 --batch 16 --pool 64 --seed 9 --layers 4 --dim 128 --out artifacts/model-energy-smoke-2`
- the train subcommand's own log: `docs/runs/2026-09-19-1.md`

| step | loss |
|---:|---:|
| 1 | 3.972e-1 |
| 100 | 4.956e-1 |
| 200 | 5.664e-1 |
| 300 | 3.374e-1 |

Finite throughout (run 1 of this smoke went NaN by step 100: `2026-09-18-energy-smoke.md`). Criterion met.

Observation for task 2: the loss sits at ~0.4 vs ~1e-3 for the same model without the energy projection — the bound `E_prev·(1 − c·ν·dt)` with c = 1 is far tighter than the solver's real per-tick dissipation, so the projection strips kinetic energy every step and the model cannot match targets. Task 2 must measure the solver's E_{t+1}/E_t on the training distribution and set the bound from it (with a small margin) before training model20.
