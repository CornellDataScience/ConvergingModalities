# Cross-modal structure experiment

Latest stage: [week-8 report](docs/week8/RESULTS.md),
[locked protocol](docs/WEEK8_PROTOCOL.md), and [reproduction guide](docs/WEEK8_RUN_GUIDE.md).
See [validation notes](docs/week8/VALIDATION.md) for template stability and transport-calibration limits.
The earlier pilot commands and report below remain preserved.

Compare independently pretrained text-only, image-only, and audio-only encoders.
Encoders stay frozen. All current outputs are exploratory. Start with
[actual results](docs/RESULTS.md), [data format](docs/DATA_GUIDE.md), and
[methods](docs/METHODS.md). The code uses small functions, descriptive names,
explicit arrays, and comments explaining scientific assumptions.

## Setup

Run from `sz684/`, using the existing virtual environment and base requirements:

```sh
uv pip install --python .venv/bin/python -r experiments/multimodal_structure/requirements.txt
.venv/bin/python -m pytest experiments/multimodal_structure/tests -q
.venv/bin/python -m experiments.multimodal_structure.run --help
```

No models or datasets download on import. Downloads, extraction, fitting,
evaluation, and reporting are explicit commands. Results and large assets are
ignored by Git. Original experiment artifacts are never overwritten by import.

## Reuse existing image–text features

The output folder must not already exist. The import preserves old selected-layer
features and weak labels. It uses original noun group 0 for test, 1 for validation,
and the remaining groups for training; it does not recreate the old five-fold run.

```sh
.venv/bin/python -m experiments.multimodal_structure.run import-legacy \
  experiments/state_transfer/results experiments/multimodal_structure/results/legacy_pilot
```

## Download a small real audiovisual dataset

```sh
.venv/bin/python -m experiments.multimodal_structure.run download-av \
  experiments/multimodal_structure/results/av_pilot --concepts 24 --examples 3
```

This samples pinned VGGSounder annotations and underlying public media. Download
logs record selection, hashes, revisions and failures. It excludes narration,
background music, static images and clips with multiple AV labels. Existing media
files can resume; a finished manifest is protected. Video-level labels do not
independently verify the chosen midpoint frame. The resulting development split
is not a fresh confirmatory benchmark.

## Extract models

`configs/models.json` is a template. Candidate entries intentionally fail until
immutable revisions, local weight hashes and training provenance are supplied.
Actual pinned specs used here are in `results/models.local.json` (machine-local
paths). The reusable default settings are documented in `configs/pilot.json`;
CLI options are explicit, rather than implicitly loading that file.

```sh
.venv/bin/python -m experiments.multimodal_structure.run extract \
  experiments/multimodal_structure/results/av_pilot/manifest.json \
  experiments/multimodal_structure/results/models.local.json beats_iter1 \
  experiments/multimodal_structure/results/av_pilot/features/beats_iter1_original.npz \
  --device cpu --batch-size 1
```

Repeat for `bloomz-560m`, `bloomz-1b7`, `dinov2_small`, and `dinov2_base` using
MPS if available. Run sequentially on a unified-memory Mac. Text/image weights
reuse the historical pinned cache. BEATs uses separately downloaded official
source and an iter1 checkpoint mirror, with recorded provenance. The adapter
rejects fine-tuned BEATs and verifies source/weight hashes. It does not claim that
a mirror checksum independently proves equivalence to the official download.

Extraction saves atomic batches and a final feature file. Changed inputs, source,
model, pooling, device, batch settings or versions invalidate resume. Use a new
output path when intentionally changing them; never delete an identity lock just
to force compatibility. Features have shape `[examples, coordinates]`; sample
IDs and dataset hashes must match, not merely row counts.

## Evaluate geometry and directions

Repeat `--cache NAME=PATH` to add models. Every pair uses one common concept
roster. Geometry supports different widths; directional fitting uses a declared
common PCA width. Both map directions are evaluated. Sparse relations are logged
as skipped, rather than receiving invented labels.

```sh
.venv/bin/python -m experiments.multimodal_structure.run evaluate \
  experiments/multimodal_structure/results/legacy_pilot/manifest.json \
  experiments/multimodal_structure/results/legacy_pilot/geometry.json \
  --cache bloomz=experiments/multimodal_structure/results/legacy_pilot/features/text.npz \
  --cache dinov2=experiments/multimodal_structure/results/legacy_pilot/features/vision.npz

.venv/bin/python -m experiments.multimodal_structure.run directions \
  experiments/multimodal_structure/results/legacy_pilot/manifest.json \
  experiments/multimodal_structure/results/legacy_pilot/directions.json \
  --cache bloomz=experiments/multimodal_structure/results/legacy_pilot/features/text.npz \
  --cache dinov2=experiments/multimodal_structure/results/legacy_pilot/features/vision.npz \
  --dimensions 8

.venv/bin/python -m experiments.multimodal_structure.run report \
  experiments/multimodal_structure/results/legacy_pilot/geometry.json \
  experiments/multimodal_structure/results/legacy_pilot/REPORT.md
```

For the new AV pilot, substitute its manifest and each model's feature cache.
It has event labels but no verified relation endpoints: geometry is supported;
directional experiments require an additional independently justified relation panel.

## Your own data

```sh
.venv/bin/python -m experiments.multimodal_structure.run prepare records.csv manifest.json
.venv/bin/python -m experiments.multimodal_structure.run import-av clips.csv new_media_folder
```

`import-av` requires system `ffmpeg` and `ffprobe`. The public downloader instead
uses the bundled executable from `imageio-ffmpeg`.

## Organization

- `data/`: validation, source/family splitting, legacy import, audiovisual import,
  public download, waveform decoding.
- `extraction/`: explicit text/image/BEATs adapters and per-model batch caches.
- `training/`: train-only PCA, balanced ridge CAV probes, Procrustes maps.
- `evaluation/`: RSA, CKA, neighbors, shuffles, CAV transfer and retrieval.
- `reporting/`: Markdown report generation from saved JSON, without refitting.
- `tests/`: analytic checks and a synthetic three-modality integration fixture.
- `results/`: local media, features, checkpoint/source assets, fits and scores.

Synthetic test results are software checks, never empirical model findings.
