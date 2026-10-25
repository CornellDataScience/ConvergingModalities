# Anisotropy Controls for Cross-Modal Alignment

How much of the measured agreement between vision and language representations is real shared
structure, and how much is an artifact of the *shape* of each embedding space? This study tests
whether preprocessing aimed at anisotropy changes mutual k-NN alignment between BLOOMZ
(language) and DINOv2 (vision) features.

The pipeline is self-contained: it builds its own inputs, runs the experiment, checks its own
baseline, and ships with tests that run on synthetic data in seconds.

## Motivation

Neural embedding spaces are rarely isotropic. Most vectors share a large common offset and a few
high-variance "rogue" dimensions, so the data occupies a narrow cone instead of the whole space.
In such a space, nearest neighbors are partly decided by the shared offset and by a handful of
dominant dimensions, not by semantics. Alignment metrics that rely on geometry (mutual k-NN, CKA,
Gromov-Wasserstein) can therefore be inflated, suppressed, or distorted. Before reading a high or
low alignment score as evidence for or against convergence between modalities, it helps to know
how sensitive that score is to anisotropy.

## What is measured

**Mutual k-NN alignment.** For paired samples (an image and its caption), find each sample's k=10
nearest neighbors in the language space and in the vision space. The score is the mean fraction of
neighbors the two spaces share. Chance overlap is about k/(N-1), so every score is reported with a
**shuffled-correspondence null**: the same overlap computed after randomly permuting which image
goes with which caption (1,000 permutations).

**Anisotropy diagnostics**, computed on the evaluation vectors after each control:

| Diagnostic | Meaning |
|---|---|
| Mean pairwise cosine | About 0 for an isotropic space, near 1 for a narrow cone |
| Effective rank | exp of the entropy of the normalized singular values of the centered data |
| Top-dimension variance share | Share of total variance in the single highest-variance raw dimension (rogue dimension) |
| Top-PC variance share | Share of total variance in the first principal component |

## Experimental design

- **Data:** 1,024 image-caption pairs from WIT.
- **Models:** BLOOMZ 560M, 1.1B, 1.7B (language) and DINOv2 small and base (vision), giving six
  model pairs.
- **Features:** one vector per layer (mean-pooled token states for BLOOMZ, the CLS token of every
  block for DINOv2). Each model's features are clipped at the 95th percentile of absolute values
  and L2-normalized.
- **Held-out layer selection:** the 1,024 pairs are split 512/512 at random, three times. The best
  layer pair is chosen on the selection half, frozen, and scored on the evaluation half. Clipping
  thresholds and every preprocessing control are fitted on the selection half only.
- **Matched nulls:** every condition uses the same 1,000 permutations, so scores are directly
  comparable across conditions.

### Controls

| Condition | What it does |
|---|---|
| `baseline` | Clipped, L2-normalized features, unchanged |
| `center` | Subtract the selection-set mean, renormalize |
| `standardize` | Center, divide each dimension by its selection-set standard deviation (deflates rogue dimensions) |
| `remove3` | Center, project out the top 3 principal directions (all-but-the-top) |
| `pca128` | Center, project onto the top 128 principal components |
| `pca128_whiten` | `pca128`, then scale each component to unit variance |

Whitening is done inside the 128-dimensional PCA subspace, because a covariance estimated from 512
rows is rank deficient for features of up to 2,048 dimensions. The parameters (128 components, 3
removed directions) were fixed in `ANISOTROPY_PROTOCOL.md` before running and not tuned.

## Findings (exploratory)

| Condition | Mean change vs baseline | Increases / 18 | Mean cosine (language / vision) |
|---|---:|---:|---|
| baseline | 0 | n/a | 0.364 / 0.476 |
| center | +0.0122 | 18 | 0.002 / 0.004 |
| standardize | +0.0118 | 18 | 0.002 / 0.004 |
| pca128 | +0.0098 | 17 | 0.002 / 0.004 |
| remove3 | +0.0070 | 14 | 0.002 / 0.003 |
| pca128_whiten | -0.0047 | 7 | 0.002 / 0.003 |

Raw baseline scores range from about 0.11 to 0.15, and every score in every condition is above its
shuffled null (108/108).

- **The spaces are strongly cone-shaped.** At baseline the mean pairwise cosine is 0.36 for
  language and 0.48 for vision.
- **Removing the shared offset helps consistently.** Centering raises alignment in all 18
  comparisons, by about 0.012, roughly a tenth of the score. Standardization gives nearly the same
  gain, which suggests the rogue dimensions were already largely tamed by clipping.
- **Going beyond centering does not help.** Removing the top 3 directions on top of centering
  lowers scores (3 of 18 increases), and whitening on top of PCA lowers them (1 of 18). The
  alignment signal appears to sit in the high-variance directions.
- **The model-size trend is fragile.** The number of strict increases across BLOOMZ sizes changes
  from condition to condition, so conclusions about scaling depend on preprocessing.

## State-transfer preprocessing study

A second experiment asks whether the same preprocessing choices change text-to-image **state
transfer** (learning "raw to cooked" from text about some objects, then using it to retrieve images
of held-out objects). It uses its own MIT-States sample and independent code.

| Path | Purpose |
|---|---|
| `STATE_CONTROLS_PROTOCOL.md` | design, six preprocessing conditions, sanity gate, limits |
| `states_data.py` | image selection and feature-cache I/O |
| `states_extract.py` | selects images from the MIT-States archive and extracts features |
| `state_transfer.py` | the experiment (numpy only) |
| `tests/test_state_transfer.py` | tests on synthetic data |

```sh
# Download the MIT-States release archive first (see the dataset page), then:
python states_extract.py --zip path/to/release_dataset.zip --run data/states --device cpu
python state_transfer.py --run data/states --out results/state_controls
```

Feature extraction for about 1,000 images and 500 prompts is resumable; DINOv2 on CPU is the slow
step. Results are exploratory (see the protocol).

## Limitations

- The 18 comparisons share models and use overlapping splits, so they are not independent
  replications. Reported ranges are sensitivity ranges, not confidence intervals, and no
  significance claims are made.
- Layers are frozen at the baseline-selected choice, so controls that favor other layers are
  penalized.
- Whitening at 128 dimensions and removing 3 directions are single fixed settings. They do not show
  that whitening or principal-component removal fails in general.
- Features come from a CPU run. Numbers will differ slightly from implementations on other
  hardware or precision, so compare trends, not decimals.
- The dataset (1,024 pairs) is small, and the controls are fitted on only 512 of them.
- The protocol was written before the run but committed afterward.

## Repository layout

| Path | Purpose |
|---|---|
| `prepare_data.py` | Builds the inputs: cached per-layer features and held-out layer selection |
| `anisotropy.py` | The experiment: controls, diagnostics, matched nulls, report |
| `ANISOTROPY_PROTOCOL.md` | Frozen protocol: conditions, parameters, reporting rules, limits |
| `tests/test_anisotropy.py` | Unit tests for controls, diagnostics and the overlap/null helpers |
| `tests/test_pipeline.py` | Synthetic end-to-end test of `prepare_data.py` followed by `anisotropy.py` |
| `reports/anisotropy-test-1/` | Snapshot of the first full report and plot |
| `requirements.txt`, `requirements-extract.txt` | Dependencies for the experiment, and for feature extraction |

`data/` and `results/` are generated and gitignored.

## Setup

```sh
conda create -n prh python=3.12 -y
conda activate prh
pip install -r requirements-extract.txt
python -m pytest -q
```

The tests need no data or network. They should all pass before you start the long extraction step.

## Run

```sh
# 1. Cache per-layer features for 5 models x 1,024 WIT pairs.
#    Downloads about 15 GB of models; slow on CPU; safe to stop and rerun (resumes).
python prepare_data.py extract --out data/baseline --device cpu

# 2. Build the splits, clipping thresholds, and held-out layer selection.
python prepare_data.py heldout --baseline data/baseline --out data/heldout_layers/results.json

# 3. Run the experiment.
python anisotropy.py
```

To keep model downloads off your system drive, set `HF_HOME` before step 1
(PowerShell: `$env:HF_HOME = "D:\hf"`; bash: `export HF_HOME=/path/to/hf`).

### Outputs (`results/anisotropy/`)

| File | Contents |
|---|---|
| `REPORT.md` | Summary table, matched differences, and interpretation limits |
| `scores.csv` | One row per split, model pair and condition |
| `results.json` | Full permutation draws plus source-file hashes for provenance |
| `anisotropy.png` | Alignment by control (left) and mean cosine after each control (right) |

### Built-in checks

`anisotropy.py` asserts that the baseline scores and all 1,000 null draws it recomputes match the
held-out file, so a bug in either script is caught rather than silently producing numbers. Results
also record SHA-256 hashes of the code, protocol and held-out file used.

## Next steps

- Quantify the centering effect with a paired bootstrap over query samples, using a separate
  reference bank, to get proper uncertainty.
- Reselect layers separately for each control to test whether the whitening result is an artifact
  of frozen layers.
- Repeat the controls with CKA and Gromov-Wasserstein-based measures, not only mutual k-NN.
- Test larger sample sizes, which needs a larger paired dataset than 1,024 pairs.
- Add further model families and sizes.

## Metric sensitivity

`metric_controls.py` re-scores the anisotropy conditions with linear CKA, Gromov-Wasserstein structure cost, and (optionally, `--gw-match`) GW blind matching, using `extra_metrics.py`. Protocol: `METRIC_CONTROLS_PROTOCOL.md`. `state_compare.py` compares the state-transfer gain across preprocessing conditions.

    python metric_controls.py --gw-match