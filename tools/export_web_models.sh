#!/usr/bin/env bash
# Export checkpoints into web/model/<name>/ for the browser gallery, plus a
# web/model/index.json the page reads to build the model picker. Weights are
# never committed (web/model/ is gitignored); this script just copies from
# artifacts/ and regenerates the index every run.
#
# Numbers below are taken from the research log (docs/runs/), not measured
# here — this script does not build or run anything.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p web/model
rm -rf web/model/*/
rm -f web/model/meta.json web/model/model.bin

copy() {
  local name="$1"
  mkdir -p "web/model/$name"
  cp "artifacts/$name/meta.json" "web/model/$name/meta.json"
  cp "artifacts/$name/model.bin" "web/model/$name/model.bin"
  echo "exported $name"
}

for name in model19 model20 model-wide-1 model-wide-2 model-energy-smoke-2 model-sph-smoke; do
  copy "$name"
done

# reverse checkpoint for the rewind exhibit, if one exists (C7, currently
# blocked on training per docs/runs/2026-09-19-rewind.md) — optional, the
# page's rewind() no-ops without it.
if [ -f artifacts/model20-reverse/meta.json ]; then
  cp artifacts/model20-reverse/meta.json web/model/model20/meta-reverse.json
  cp artifacts/model20-reverse/model.bin web/model/model20/model-reverse.bin
  echo "exported model20 reverse checkpoint"
fi

cat > web/model/index.json <<'EOF'
[
  {
    "name": "model20",
    "dir": "model20",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "recipe": "700 steps, lr 1e-4, rollout 8, nu 0.001, mass+energy projection ON (r_max 1.0001), warm start from model19 (docs/runs/2026-09-19-2.md)",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9997,
    "corr25": 0.9988,
    "corr50": 0.9954,
    "amp50": 1.0329,
    "lag50": -0.007,
    "energyViolation": 0.0,
    "note": "BEST per docs/runs/2026-09-19-rung6.md — best corr at every horizon, smallest 50-tick wavefront lag, only checkpoint trained with both projections on."
  },
  {
    "name": "model19",
    "dir": "model19",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "recipe": "700 steps, lr 1e-4, rollout 8, nu 0.001, no projection, warm start from model17, 70/30 standard/extended mixture (docs/runs/2026-09-18-17.md)",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9995,
    "corr25": 0.9980,
    "corr50": 0.9927,
    "amp50": 1.0460,
    "lag50": -0.209,
    "energyViolation": null,
    "note": "model20's parent checkpoint; no mass/energy projection during training."
  },
  {
    "name": "model-wide-1",
    "dir": "model-wide-1",
    "params": 5427971,
    "dim": 256,
    "layers": 6,
    "mask": "stencil r1 d1",
    "recipe": "3000 steps, lr 3e-4, rollout 4, nu 0.001, mass projection only, from scratch, 3080 box (docs/runs/2026-09-19-rung6.md)",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9987,
    "corr25": 0.9956,
    "corr50": 0.9858,
    "amp50": 1.0368,
    "lag50": 0.186,
    "energyViolation": 0.0,
    "note": "5.5x model20's params; does not beat model20 on this eval despite the width."
  },
  {
    "name": "model-wide-2",
    "dir": "model-wide-2",
    "params": 5427971,
    "dim": 256,
    "layers": 6,
    "mask": "stencil r1 d1",
    "recipe": "3000 more steps, lr 2e-4, rollout 8, nu 0.001, mass projection only, warm start from model-wide-1, 3080 box (docs/runs/2026-09-19-rung6.md)",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9986,
    "corr25": 0.9935,
    "corr50": 0.9760,
    "amp50": 1.0467,
    "lag50": -0.026,
    "energyViolation": 0.0,
    "note": "rollout-8 continuation of model-wide-1; scores slightly worse (mild overfit), still passes all four acceptance bars."
  },
  {
    "name": "model-energy-smoke-2",
    "dir": "model-energy-smoke-2",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "recipe": "300-step smoke, lr 1e-4, rollout 4, nu 0.001, mass+energy projection ON (bound not yet fit to the solver), warm start from model19 (docs/runs/2026-09-19-energy-smoke-2.md)",
    "kind": "eulerian",
    "loadable": true,
    "corr10": null,
    "corr25": null,
    "corr50": null,
    "amp50": null,
    "lag50": null,
    "energyViolation": null,
    "note": "smoke run only (finite-loss check after the projection-detach fix), no eval-metrics pass was run against this checkpoint; loss ~0.4 vs ~1e-3 unprojected because the bound was too tight (superseded by model20's fitted r_max)."
  },
  {
    "name": "model-sph-smoke",
    "dir": "model-sph-smoke",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "recipe": "SPH-splatted (Lagrangian) smoke checkpoint, exact train command/step count not recoverable (no run doc found, docs/runs/2026-09-19-sph-smoke.md)",
    "kind": "lagrangian-splat",
    "loadable": true,
    "corr10": null,
    "corr25": 0.36,
    "corr50": null,
    "amp50": null,
    "lag50": null,
    "energyViolation": null,
    "note": "valid only on the SPH cases (dam break, drop, both SPH-splatted): corr@25 0.35 (dam break) / 0.37 (drop), amplitude ratio 9.7x/15.4x — under-trained, not a bug. Do not run on the shallow-water cases."
  }
]
EOF
echo "wrote web/model/index.json"
