#!/usr/bin/env bash
# Model size x excess-energy constraint x seed, matched budget
# (docs/runs/2026-10-06-energy-band.md). Run from the repo root on the RTX 3080 machine:
#   DATASET=<dump of `jacobi dump-dataset --pool 1000 --nu 0.001 --seed 20`> \
#   PYTHON=<venv python with torch> JACOBI=<wgpu build of the cli> \
#   bash tools/energy-band-sweep.sh
# Resumable: finished trainings, imports and evals are skipped.
set -euo pipefail

DATASET=${DATASET:?dump-dataset directory}
PYTHON=${PYTHON:-python}
JACOBI=${JACOBI:-./target-wgpu/release/jacobi}
OUT=${OUT:-results/band}
STEPS=${STEPS:-3000}
TRAJ=${TRAJ:-24}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
DATASET=$(cd "$DATASET" && pwd)

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU busy, not starting" >&2
    exit 1
fi

# Per-tick ratio of the cross-term excess energy at nu = 0.001 on the
# training mixture: [0.9998135, 1.0003122] over 39,411 dumped ticks,
# [0.9997747, 1.0003457] over 51,140 generator ticks (energy-stats).
R_MIN=0.99975
R_MAX=1.00036

# model-wide-1's recipe (docs/runs/2026-09-19-3.md, the rung6 width comparison),
# from scratch, for both sizes; train.py's defaults already match it.
COMMON=(--dataset "$DATASET" --steps "$STEPS" --lr 3e-4 --batch 16 --rollout 4
    --rollout-after 200 --noise 0.04 --project true --project-energy false
    --spectral 1 --lr-floor 1)

arm_flags() {
    case "$1" in
        off) ;;
        upper) echo "--excess-r-max $R_MAX --excess-r-min 0" ;;
        band) echo "--excess-r-max $R_MAX --excess-r-min $R_MIN" ;;
    esac
}

inference_flags() {
    case "$1" in
        off) echo "--project-excess 0" ;;
        upper) echo "--project-excess $R_MAX --excess-r-min 0" ;;
        band) echo "--project-excess $R_MAX --excess-r-min $R_MIN" ;;
    esac
}

RUNS=()
for seed in 20 21 22; do
    for size in "128 4" "256 6"; do
        read -r dim layers <<< "$size"
        for arm in off upper band; do
            RUNS+=("d${dim}l${layers}-s${seed}-${arm}")
        done
    done
done

for name in "${RUNS[@]}"; do
    [[ "$name" =~ ^d([0-9]+)l([0-9]+)-s([0-9]+)-([a-z]+)$ ]]
    dim=${BASH_REMATCH[1]} layers=${BASH_REMATCH[2]} seed=${BASH_REMATCH[3]} arm=${BASH_REMATCH[4]}
    if [[ ! -f "$OUT/$name/config.json" ]]; then
        echo "$(date '+%FT%T') train $name"
        # shellcheck disable=SC2046
        (cd training && "$PYTHON" -m jacobi2000_pt.train "${COMMON[@]}" --dim "$dim" \
            --layers "$layers" --seed "$seed" $(arm_flags "$arm") --out "$OUT/$name" \
            --log-file "$OUT/$name-log.json") > "$OUT/$name.train.log" 2>&1
    fi
    if [[ ! -f "$OUT/$name-burn/meta.json" ]]; then
        "$JACOBI" import-weights --safetensors "$OUT/$name/model.safetensors" \
            --config "$OUT/$name/config.json" --out "$OUT/$name-burn" > /dev/null
    fi
done

for name in "${RUNS[@]}"; do
    arm=${name##*-}
    case "$arm" in
        off) inferences=(off upper band) ;;
        *) inferences=(off "$arm") ;;
    esac
    for inf in "${inferences[@]}"; do
        f="$OUT/$name.eval-$inf.txt"
        [[ -s "$f" ]] && grep -q '^| model | 50 |' "$f" && continue
        echo "$(date '+%FT%T') eval $name inference $inf"
        # shellcheck disable=SC2046
        "$JACOBI" eval-metrics --model "$OUT/$name-burn" --horizons 10,25,50 --trajectories "$TRAJ" \
            --seed 0 --project true --project-energy false $(inference_flags "$inf") > "$f" 2>&1
    done
done

{
    echo "| run | inference | ticks | corr (h,u,v) | corr (h) | active RMSE | amplitude ratio | radius err | lag | band 2-4 | band 4-8 | band 8-16 | global RMSE | bias |"
    echo "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for f in "$OUT"/*.eval-*.txt; do
        b=$(basename "$f" .txt)
        grep -E '^\| model \| (10|25|50) \|' "$f" | sed "s/^| model /| ${b%%.eval-*} | ${b##*.eval-} /"
    done
    echo
    echo "| run | inference | E' ON / E'_0 @50 | E' OFF / E'_0 @50 | solver @50 | active ticks |"
    echo "|---|---|---:|---:|---:|---|"
    for f in "$OUT"/*.eval-*.txt; do
        b=$(basename "$f" .txt)
        # The E' section exists only when a projection is on (absent from eval-off).
        row=$( (grep -E '^\| 50 \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \|$' "$f" || true) | tail -1 | cut -d'|' -f3-5)
        act=$(grep -oE '[0-9]+ of [0-9]+ ticks \([0-9.]+ %\)' "$f" || true)
        if [[ -n "$row" ]]; then
            echo "| ${b%%.eval-*} | ${b##*.eval-} |$row| $act |"
        fi
    done
} > "$OUT/summary.md"
echo "$(date '+%FT%T') done"
