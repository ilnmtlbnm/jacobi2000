#!/usr/bin/env bash
# Copy checkpoints into web/model/<name>/ for the browser demo and write
# web/model/index.json, which the page reads to build its model picker.
# A checkpoint is a directory holding model.safetensors and config.json, as
# written by training/jacobi2000_pt/train.py or downloaded with `hf download`.
# web/model/ is gitignored and rebuilt on every run.
#
#   tools/export_web_models.sh model20=models/model20 [name=dir ...]
#
# The first checkpoint is the one the page opens with. If <dir>-reverse holds
# a reverse-time checkpoint, it is copied next to it for the rewind exhibit.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  echo "usage: $0 name=checkpoint-dir [name=checkpoint-dir ...]" >&2
  exit 2
fi

rm -rf web/model
mkdir -p web/model

for pair in "$@"; do
  name="${pair%%=*}"
  src="${pair#*=}"
  [ -f "$src/model.safetensors" ] && [ -f "$src/config.json" ] || { echo "$src: need model.safetensors and config.json" >&2; exit 1; }
  mkdir -p "web/model/$name"
  cp "$src/model.safetensors" "$src/config.json" "web/model/$name/"
  if [ -f "$src-reverse/model.safetensors" ]; then
    cp "$src-reverse/model.safetensors" "web/model/$name/model-reverse.safetensors"
    cp "$src-reverse/config.json" "web/model/$name/config-reverse.json"
  fi
  echo "copied $name <- $src"
done

python3 - "$@" <<'PY'
import json, struct, sys

entries = []
for pair in sys.argv[1:]:
    name = pair.split("=", 1)[0]
    d = f"web/model/{name}"
    cfg = json.load(open(f"{d}/config.json"))
    with open(f"{d}/model.safetensors", "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    params = 0
    for key, t in header.items():
        if key != "__metadata__":
            n = 1
            for s in t["shape"]:
                n *= s
            params += n
    mask = cfg["mask"]
    mask = "full attention" if mask == "Full" else "stencil r{radius} d{dilation}".format(**mask["Stencil"])
    entries.append({
        "name": name, "dir": name, "params": params, "dim": cfg["dim"], "layers": cfg["layers"],
        "mask": mask, "desc": mask, "kind": "eulerian", "loadable": True,
        "corr10": None, "corr25": None, "corr50": None, "amp50": None, "lag50": None,
        "energyViolation": None,
    })
json.dump(entries, open("web/model/index.json", "w"), indent=2)
print("wrote web/model/index.json")
PY
