# Anisotropy Controls for Cross-Modal Alignment

Self-contained study of how anisotropy-targeting preprocessing (standardization, top-PC
removal, whitened PCA) changes mutual k-NN alignment between BLOOMZ and DINOv2.
Nothing here imports or copies code from any other repository.

| File | Purpose |
|---|---|
| `prepare_data.py` | builds the inputs: cached per-layer features + held-out layer selection |
| `anisotropy.py` | the experiment |
| `ANISOTROPY_PROTOCOL.md` | frozen protocol; commit before running the experiment |
| `test_anisotropy.py`, `test_pipeline.py` | unit tests and a synthetic end-to-end test |

## Setup

```sh
conda create -n prh python=3.12 -y && conda activate prh
pip install -r requirements-extract.txt
python -m pytest -q
```

## Run

```sh
# 1. features for 5 models x 1024 WIT pairs (downloads ~15 GB; slow on CPU, resumable)
python prepare_data.py extract --out data/baseline --device cpu
# 2. splits, clipping thresholds, held-out layer selection
python prepare_data.py heldout --baseline data/baseline --out data/heldout_layers/results.json
# 3. the anisotropy experiment -> results/anisotropy/REPORT.md
python anisotropy.py
```

Add `data/` and `results/` to `.gitignore`. Set the `HF_HOME` environment variable first to keep
model downloads off your system drive. `anisotropy.py` asserts that the baseline scores and null
draws it recomputes match the held-out file, so a bug in either script is caught.

The numbers will not exactly match any other implementation (different hardware and precision);
compare trends, not decimals.