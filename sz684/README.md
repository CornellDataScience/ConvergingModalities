# Cross-modal representation experiments

The latest completed study is **[natural object-state transfer](experiments/state_transfer/)**, including raw → cooked. Each experiment now owns its code, protocols, tests, and saved results.

## Experiments

| Folder | Question / status |
| --- | --- |
| [state_transfer](experiments/state_transfer/) | **Latest:** can text-derived state changes transfer to images of held-out nouns? |
| [difference_transfer](experiments/difference_transfer/) | Earlier supervised arbitrary difference-vector transfer |
| [endpoint_audit](experiments/endpoint_audit/) | Earlier endpoint-contribution audit and vector explorer |
| [baseline](experiments/baseline/) | Earlier WIT alignment baseline and language evaluation |
| [heldout_layers](experiments/heldout_layers/) | Earlier held-out layer-selection study |
| [robustness](experiments/robustness/) | Earlier sample-size and preprocessing sensitivity |
| [paired_uncertainty](experiments/paired_uncertainty/) | Earlier conditional paired alignment inference |
| [color_transfer](experiments/color_transfer/) | **Retired:** Quick, Draw! red → blue study; main test scoring was not completed |

## Folder roles

Each experiment includes only the roles it needs:

```text
experiments/<experiment>/
  README.md     purpose, status, commands, and file map
  data/         acquisition, dataset manifests, frozen feature extraction
  training/     supervised translator fitting
  evaluation/   validation, retrieval/scoring, controls, uncertainty
  reporting/    report, figure, and PDF generation
  configs/      fixed experiment settings
  docs/         protocols, manuscripts, and notes
  tests/        experiment-specific checks
  results/      saved datasets, features, scores, locks, snapshots (Git-ignored)
  output/       rendered figures and PDFs, where applicable
shared/         reusable transformations, ridge solver, I/O, paths, cache validation
```

Encoder feature extraction is a data transformation. BLOOMZ and DINOv2 weights stay frozen. The state translator is trained in [training/bridge.py](experiments/state_transfer/training/bridge.py); its ridge solver is [shared/training.py](shared/training.py). Earlier experiments call that same solver directly from their evaluation pipelines.

## Running experiments

Use Python 3.12 and the recorded environment from `sz684/`:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python run.py --list
```

For the latest study, see [its README](experiments/state_transfer/README.md). Example commands:

```sh
.venv/bin/python run.py state-data
.venv/bin/python run.py state-extract --modality language --device mps
.venv/bin/python run.py state-extract --modality vision --device mps
.venv/bin/python run.py state-evaluate
.venv/bin/python run.py state-followup
.venv/bin/python run.py state-report
```

The runner can also be launched by absolute path from another directory. CLI paths resolve relative to `sz684/`; absolute paths are accepted. `run.py` sets a stable working directory for older report code. Pass a stage followed by `--help` to see its arguments. Evaluation scripts without arguments begin their analysis directly.

Feature extraction uses macOS MPS, pinned cached weights, and resumable batches; run modalities sequentially. The state dataset builder also requires the earlier baseline manifest for pinned revisions. Neither cached weights nor saved run artifacts are tracked by Git; the [baseline setup guide](experiments/baseline/docs/LEGACY_BASELINE.md) documents that dependency.

## Latest findings and provenance

Across 992 MIT-States images, 51 noun labels, and 496 queries, joint state-and-noun top-1 is 38.7% with the text direction, 20.9% for source similarity, and 42.1% for direct target text. All four planned AP contrasts are positive, but noun retention falls from 86.6% to 79.4%; the full success criterion fails. These are independent photos with weak released labels and noun-label holdout, not verified physical transitions or taxonomic exclusion.

Read the [accessible manuscript](experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md), [technical manuscript](experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md), and [data audit](experiments/state_transfer/docs/STATE_DATA_AUDIT.md). [CODE_GUIDE.md](CODE_GUIDE.md) explains the implementation; [CURRENT_DIRECTION.md](CURRENT_DIRECTION.md) records the research status. Older cross-experiment reviews live in [docs/history/](docs/history/).

Saved run directories moved with their experiments; their contents, locks, metadata, and source snapshots are preserved. Historical paths are resolved by [shared/paths.py](shared/paths.py). The two relocated extractors accept their specifically recorded pre-move cache identities only when all settings match and the migrated extractor source is unchanged. Unknown source hashes or changed extraction settings still fail validation.

Analysis/report commands replace their own generated outputs when rerun. Preserve historical results before intentionally regenerating them. Working-source hashes differ from the original run; frozen snapshots remain the source of truth for that run.
