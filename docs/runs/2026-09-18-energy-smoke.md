# Task 11 — energy-projection smoke run (FAILED: diverged)

Machine: a 16 GB consumer machine. Backend: wgpu (Metal). GPU uncontended
(`ps -A -o comm | grep -E 'target/release/(jacobi|llm-life)'` empty before
launch). Commit at launch: `053c6a3`. Warm start: `artifacts/model19`
(BEST, task 8c).

Command:

```
cargo run --release -p cli --features wgpu --no-default-features -- train \
  --init artifacts/model19 --mask stencil --radius 1 --dilation 1 \
  --steps 300 --single-drop true --project true --project-energy true \
  --nu 0.001 --rollout 4 --rollout-after 0 --noise 0.04 --spectral 1 \
  --lr 0.0001 --batch 16 --pool 64 --seed 8 --layers 4 --dim 128 \
  --out artifacts/model-energy-smoke \
  --note "task 11 smoke: mass+energy projection"
```

This run enables both projections together for the first time (`--project
true --project-energy true`) on a checkpoint that has only ever been
trained/warm-started with mass projection alone.

## Outcome: diverged, process killed

The `train` subcommand does not write its own run-doc file when it exits
abnormally (it did not exit normally here), so this doc is written by hand
from the raw stdout log. No checkpoint was
written to `artifacts/model-energy-smoke` (the run never reached a save
point).

| step | loss | wall clock |
|---:|---:|---:|
| 1 | 5.019586e-1 | 2.0s |
| 100 | NaN | 186.5s |

Loss is already NaN by step 100. The run was left running past this point
to see whether it recovered; it did not — the process was killed
(~step 150, no further log line printed before then since `log_every =
100` means the next print would have been step 200) rather than let it run
to no purpose. **This smoke test fails task 11's convergence criterion
("finite, non-NaN loss") outright.** Task 11 stays open.

## Divergence hypothesis

`crates/jacobi2000/src/project.rs::project_energy` computes, per batch
member:

```rust
let budget = (bound - potential).clamp_min(0.0);
let scale = (budget / kinetic.clamp_min(1e-12)).sqrt().clamp(0.0, 1.0);
```

`sqrt(x)` has an unbounded derivative as `x -> 0`
(`d/dx sqrt(x) = 1/(2*sqrt(x)) -> inf`). Whenever a batch member's
potential energy alone already meets or exceeds the bound, `budget` is
clamped to exactly `0.0`, `scale` evaluates to exactly `0.0`, and the
backward pass through `sqrt` at that point is `inf` (or `NaN` once
multiplied through zero elsewhere in the graph). With `nu = 0.001` the
bound decays only very slightly per step (`bound = prev_energy * (1 -
c*nu*dt)`), so `budget = 0` is a plausible, not-rare state whenever the
model's own potential-energy estimate drifts even slightly above the
previous step's total — and this run also unrolls a 4-step differentiable
rollout, giving 4 chances per training step for the projection to hit that
zero and for the resulting `inf`/`NaN` gradient to propagate back into the
shared weights. The two existing tests for `project_energy` in
`crates/jacobi2000/src/project.rs`
(`energy_projection_scales_down_to_the_bound_and_leaves_mass_alone`,
`energy_projection_is_identity_within_the_bound`) only assert forward
values; neither exercises the projection inside a backward pass, so this
gradient singularity was never caught.

This is a plausible, not yet confirmed, diagnosis — no repro test was
written and no fix was attempted; the decision here was to document and
stop rather than fix in this step.

## Suggested fix (for the next attempt, not applied here)

Either:
1. **Stop-gradient / detach `scale`.** The projection is a hard
   post-hoc correction (the design notes, sec. 4, "propose, then project"), not
   something the model should be learning to anticipate through its
   gradient; detaching `scale` (and/or `budget`/`kinetic` going into it)
   from the autodiff graph removes the singular derivative entirely while
   leaving the forward-pass physics identical to today's tests.
2. **Smooth the clamp near zero**, e.g. `sqrt((budget + eps) /
   (kinetic + eps))` with a small `eps` (or a soft `clamp_min` such as a
   softplus), so the derivative stays finite as `budget -> 0`. Cheaper to
   reason about than (1) but changes the forward numerics slightly (the
   two existing tests' tolerances would need checking against a nonzero
   `eps`).

(1) is preferred: it matches the doc comment's own framing of projection as
a post-processing step outside the model's learned dynamics, and it would
need no new tolerance tuning of the existing forward-value tests.
