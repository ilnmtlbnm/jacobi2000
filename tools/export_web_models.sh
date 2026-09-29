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

for name in model19 model20 model-wide-1 model-wide-2 model-energy-smoke-2 model-sph-smoke model-pt-curriculum; do
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
    "desc": "trained with both the mass and energy conservation bounds turned on, warm-started from a smaller checkpoint without them.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9997,
    "corr25": 0.9988,
    "corr50": 0.9954,
    "amp50": 1.0329,
    "lag50": -0.007,
    "energyViolation": 0.0
  },
  {
    "name": "model19",
    "dir": "model19",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "desc": "the parent checkpoint, trained without either conservation bound.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9995,
    "corr25": 0.9980,
    "corr50": 0.9927,
    "amp50": 1.0460,
    "lag50": -0.209,
    "energyViolation": null
  },
  {
    "name": "model-wide-1",
    "dir": "model-wide-1",
    "params": 5427971,
    "dim": 256,
    "layers": 6,
    "mask": "stencil r1 d1",
    "desc": "a wider model, 5.4M parameters, trained from scratch with the mass bound only.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9987,
    "corr25": 0.9956,
    "corr50": 0.9858,
    "amp50": 1.0368,
    "lag50": 0.186,
    "energyViolation": 0.0
  },
  {
    "name": "model-wide-2",
    "dir": "model-wide-2",
    "params": 5427971,
    "dim": 256,
    "layers": 6,
    "mask": "stencil r1 d1",
    "desc": "continued training of the wide model with the mass bound only, more rollout steps.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9986,
    "corr25": 0.9935,
    "corr50": 0.9760,
    "amp50": 1.0467,
    "lag50": -0.026,
    "energyViolation": 0.0
  },
  {
    "name": "model-energy-smoke-2",
    "dir": "model-energy-smoke-2",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "desc": "a short trial run with both conservation bounds on, before the energy bound was properly calibrated.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": null,
    "corr25": null,
    "corr50": null,
    "amp50": null,
    "lag50": null,
    "energyViolation": null
  },
  {
    "name": "model-pt-curriculum",
    "dir": "model-pt-curriculum",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "desc": "trained in PyTorch with a six-stage curriculum that follows model20's recipe, then loaded here unchanged.",
    "kind": "eulerian",
    "loadable": true,
    "corr10": 0.9995,
    "corr25": 0.9981,
    "corr50": 0.9930,
    "amp50": 1.0005,
    "lag50": 0.063,
    "energyViolation": 0.0
  },
  {
    "name": "model-sph-smoke",
    "dir": "model-sph-smoke",
    "params": 995843,
    "dim": 128,
    "layers": 4,
    "mask": "stencil r1 d1",
    "desc": "trained on particle-based (SPH) data instead of the grid solver; only valid on the particle-splatted test cases.",
    "kind": "lagrangian-splat",
    "loadable": true,
    "corr10": null,
    "corr25": 0.36,
    "corr50": null,
    "amp50": null,
    "lag50": null,
    "energyViolation": null
  }
]
EOF
echo "wrote web/model/index.json"
