```
python3 -m venv .venv
. .venv/bin/activate
pip install -e training
cargo run --release -p engine-cli --bin dump-dataset -- --pool 24 --out data/ds
python -m jacobi2000_pt.train --dataset data/ds --steps 300 --out artifacts/model
cargo run -p engine-cli --release --bin rollout -- \
    --model artifacts/model --steps 50 --backend cpu
```
