# Cross-modal representation experiments

Start with the [newcomer guide and data-flow diagrams](docs/NEWCOMER_GUIDE.md), then use the [file inventory](docs/FILE_INVENTORY.md) to locate individual files and saved artifacts.

The current research plan has two state-transfer tracks: understand cross-modal state directions, and preserve object identity with a source-dependent shift. **Both are planning only; neither track is implemented.** The [work allocation and rerun checklist](docs/planning/STATE_RELATIONS_PLAN.md) describes future work.

The completed [natural object-state study](experiments/state_transfer/) supplies the reference result. The newer [multimodal structure study](experiments/multimodal_structure/) contains exploratory six-model text/image/audio results. Each experiment owns its code, protocols, tests, and saved results.

## Experiments

| Folder | Question / status |
| --- | --- |
| [state_transfer](experiments/state_transfer/) | **Completed reference:** text-derived state changes on held-out nouns; preservation follow-ups are exploratory |
| [multimodal_structure](experiments/multimodal_structure/) | **Exploratory:** frozen text/image/audio geometry and category CAV transfer; [week-8 results](experiments/multimodal_structure/docs/week8/RESULTS.md) |
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
  extraction/   separate frozen-model adapters/caches in multimodal_structure
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

For the completed state-transfer study, see [its README](experiments/state_transfer/README.md). The following are reproduction commands, not steps needed for repository cleanup:

```sh
.venv/bin/python run.py state-data
.venv/bin/python run.py state-extract --modality language --device mps
.venv/bin/python run.py state-extract --modality vision --device mps
.venv/bin/python run.py state-evaluate
.venv/bin/python run.py state-followup
.venv/bin/python run.py state-report
```

The runner can also be launched by absolute path from another directory. CLI paths resolve relative to `sz684/`; absolute paths are accepted. `run.py` sets a stable working directory for older report code. Pass a stage followed by `--help` to see its arguments. Evaluation scripts without arguments begin their analysis directly.

The multimodal experiment has its own runner: `.venv/bin/python -m experiments.multimodal_structure.run --help`. Its [run guide](experiments/multimodal_structure/docs/WEEK8_RUN_GUIDE.md) covers additional dependencies and saved fits/locks; those stages are not registered in the top-level runner.

Feature extraction uses macOS MPS, pinned cached weights, and resumable batches; run modalities sequentially. The state dataset builder also requires the earlier baseline manifest for pinned revisions. Neither cached weights nor saved run artifacts are tracked by Git; the [baseline setup guide](experiments/baseline/docs/LEGACY_BASELINE.md) documents that dependency.

## State-transfer findings and provenance

Across 992 MIT-States images, 51 noun labels, and 496 queries, joint state-and-noun top-1 is 38.7% with the text direction, 20.9% for source similarity, and 42.1% for direct target text. All four planned AP contrasts are positive, but noun retention falls from 86.6% to 79.4%; the full success criterion fails. These are independent photos with weak released labels and noun-label holdout, not verified physical transitions or taxonomic exclusion.

Read the [accessible manuscript](experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md), [technical manuscript](experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md), and [data audit](experiments/state_transfer/docs/STATE_DATA_AUDIT.md). [CODE_GUIDE.md](CODE_GUIDE.md) explains the implementation; [CURRENT_DIRECTION.md](CURRENT_DIRECTION.md) records the research status. Older cross-experiment reviews live in [docs/history/](docs/history/).

Saved run directories moved with their experiments; their contents, locks, metadata, and source snapshots are preserved. Historical paths are resolved by [shared/paths.py](shared/paths.py). The two relocated extractors accept their specifically recorded pre-move cache identities only when all settings match and the migrated extractor source is unchanged. Unknown source hashes or changed extraction settings still fail validation.

Analysis/report commands replace their own generated outputs when rerun. Preserve historical results before intentionally regenerating them. Working-source hashes differ from the original run; frozen snapshots remain the source of truth for that run.

## Documentation and cleanup

- [CURRENT_DIRECTION.md](CURRENT_DIRECTION.md): research status and the two planned tracks.
- [CODE_GUIDE.md](CODE_GUIDE.md): detailed state-transfer implementation walkthrough.
- [Newcomer guide](docs/NEWCOMER_GUIDE.md): folder map, data flow, and reading order.
- [File inventory](docs/FILE_INVENTORY.md): maintained files, dependencies, and local artifact groups.
- [Cleanup record](docs/CLEANUP_RECORD.md): removed clutter, relocated documents, and preserved dependencies.
- [Historical documents](docs/history/): earlier reviews and the superseded multimodal expansion proposal.
