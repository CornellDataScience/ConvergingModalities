# File inventory

Snapshot after the documentation cleanup. Paths are relative to `sz684/`. Every maintained file is listed individually below; all remaining files are covered by non-overlapping recursive artifact/runtime groups. Counts and byte sizes are local snapshots, not claims that ignored artifacts are distributed through Git. Byte totals are logical file sizes and follow file symlinks; they can double-count cached targets and differ from disk usage reported by `du`.

Read the [newcomer diagrams](NEWCOMER_GUIDE.md) for data flow and [cleanup record](CLEANUP_RECORD.md) for removals. Every listed existing file is retained. New research files appear only as proposals in the [future work plan](planning/STATE_RELATIONS_PLAN.md).

## Local artifact and runtime groups

| Path (recursive) | Files | Bytes | Purpose and retention reason |
| --- | ---: | ---: | --- |
| `.venv/` | 27,932 | 873,077,570 | Installed Python runtime/libraries; keep to run the recorded environment |
| `.cache/huggingface/` | 77 | 15,980,299,279 | Cached model weights/datasets or downloaded archive; keep for reproduction, not working source |
| `.cache/mit_states/` | 1 | 17,282,640 | Cached model weights/datasets or downloaded archive; keep for reproduction, not working source |
| `.cache/openwebtext-10k.tar.xz` | 1 | 14,723,792 | Cached model weights/datasets or downloaded archive; keep for reproduction, not working source |
| `.cache/quickdraw/` | 28 | 16,271,764 | Cached model weights/datasets or downloaded archive; keep for reproduction, not working source |
| `experiments/baseline/results/` | 6,271 | 5,155,279,896 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/color_transfer/results/` | 2,137 | 28,426,131 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/difference_transfer/results/` | 4 | 469,191 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/endpoint_audit/results/` | 4 | 118,834 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/heldout_layers/results/` | 4 | 771,904 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/multimodal_structure/results/` | 5,979 | 4,903,832,897 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/paired_uncertainty/results/` | 5 | 11,107,822 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/robustness/results/` | 6 | 3,651,559 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/state_transfer/results/` | 1,120 | 52,912,199 | Manifests, media, features, fits, locks, scores and source snapshots; preserve historical evidence. Multimodal assets also include downloaded encoder source/weights |
| `experiments/state_transfer/output/` | 6 | 7,378,341 | Rendered figures/PDFs; preserve alongside manuscripts and saved results |

## Maintained files

Local dependency entries below are statically detected Python imports, including relative imports resolved within each experiment package. Runtime file dependencies are summarized after the table. An em dash does not imply that a script is independent of manifests or saved data.

### Top-level and shared support

| File | Purpose | Local imports |
| --- | --- | --- |
| [CODE_GUIDE.md](../CODE_GUIDE.md) | Reading the state-transfer code | — |
| [CURRENT_DIRECTION.md](../CURRENT_DIRECTION.md) | Current research direction | — |
| [README.md](../README.md) | Cross-modal representation experiments | — |
| [pytest.ini](../pytest.ini) | Test discovery and import-path configuration | — |
| [requirements-analysis.txt](../requirements-analysis.txt) | Analysis dependency overlay | — |
| [requirements.lock.txt](../requirements.lock.txt) | Recorded base environment versions | — |
| [requirements.txt](../requirements.txt) | Base dependency constraints | — |
| [run.py](../run.py) | Command dispatcher; does not itself fit an encoder | `shared/paths.py` |
| [shared/__init__.py](../shared/__init__.py) | Python package marker | — |
| [shared/cache.py](../shared/cache.py) | Strict resumable cache identity validation | — |
| [shared/data.py](../shared/data.py) | Feature loading, train-fitted clipping/PCA and gallery masks | — |
| [shared/extractor_migrations.json](../shared/extractor_migrations.json) | Exact reorganized extractor hashes allowed for legacy cache identities | — |
| [shared/io.py](../shared/io.py) | Manifest hashing, I/O, pooling and device helpers | `shared/paths.py` |
| [shared/paths.py](../shared/paths.py) | Canonical project root and historical path aliases | — |
| [shared/tests/test_layout.py](../shared/tests/test_layout.py) | Folder migration must preserve access to historical artifacts and cache safety. | `experiments/state_transfer/training/bridge.py`<br>`shared/cache.py`<br>`shared/paths.py`<br>`shared/training.py` |
| [shared/training.py](../shared/training.py) | Centered ridge solver | — |
| [shared/transforms.py](../shared/transforms.py) | Historical fit/apply representation transforms | — |

### Documentation

| File | Purpose | Local imports |
| --- | --- | --- |
| [docs/CLEANUP_RECORD.md](../docs/CLEANUP_RECORD.md) | Exact disposable deletions, document move and preservation checks | — |
| [docs/FILE_INVENTORY.md](../docs/FILE_INVENTORY.md) | This complete maintained-file index and recursive artifact inventory | — |
| [docs/NEWCOMER_GUIDE.md](../docs/NEWCOMER_GUIDE.md) | Repository tree and three current data-flow diagrams | — |
| [docs/history/MULTIMODAL_EXPANSION_PLAN.md](../docs/history/MULTIMODAL_EXPANSION_PLAN.md) | Independently trained unimodal models: cross-modal research plan | — |
| [docs/history/RESEARCH_REPORT.md](../docs/history/RESEARCH_REPORT.md) | Cross-modal representation alignment: robustness and difference-vector transfer | — |
| [docs/history/RESEARCH_REVIEW.md](../docs/history/RESEARCH_REVIEW.md) | Research assessment and decision | — |
| [docs/planning/STATE_RELATIONS_PLAN.md](../docs/planning/STATE_RELATIONS_PLAN.md) | Planning-only owners, future files, dependencies and rerun checklist | — |

### Experiment package root

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/__init__.py](../experiments/__init__.py) | Python package marker | — |

### state_transfer — Completed reference; follow-ups exploratory

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/state_transfer/README.md](../experiments/state_transfer/README.md) | Completed reference study: natural object-state transfer | — |
| [experiments/state_transfer/__init__.py](../experiments/state_transfer/__init__.py) | Python package marker | — |
| [experiments/state_transfer/configs/state_transfer.json](../experiments/state_transfer/configs/state_transfer.json) | Fixed experiment/model settings | — |
| [experiments/state_transfer/data/__init__.py](../experiments/state_transfer/data/__init__.py) | Python package marker | — |
| [experiments/state_transfer/data/mit_states_data.py](../experiments/state_transfer/data/mit_states_data.py) | Selectively read the original MIT-States ZIP with verified HTTP byte ranges. | `shared/io.py`<br>`shared/paths.py` |
| [experiments/state_transfer/data/state_data.py](../experiments/state_transfer/data/state_data.py) | Build a five-relation natural-image benchmark from the original MIT-States release. | `experiments/state_transfer/data/mit_states_data.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/state_transfer/data/state_extract.py](../experiments/state_transfer/data/state_extract.py) | Pinned offline model extraction, unique text prompts, resumable vision batches. | `shared/cache.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/state_transfer/docs/STATE_DATA_AUDIT.md](../experiments/state_transfer/docs/STATE_DATA_AUDIT.md) | Natural-state data audit | — |
| [experiments/state_transfer/docs/STATE_FOLLOWUP_PROTOCOL.md](../experiments/state_transfer/docs/STATE_FOLLOWUP_PROTOCOL.md) | Exploratory preservation follow-up | — |
| [experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md](../experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md) | Do State-Change Directions Transfer from Text to Images? | — |
| [experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md](../experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md) | Can a Change Learned in Words Transfer to New Images? | — |
| [experiments/state_transfer/docs/STATE_TRANSFER_PROTOCOL.md](../experiments/state_transfer/docs/STATE_TRANSFER_PROTOCOL.md) | Natural-image state transfer: primary protocol | — |
| [experiments/state_transfer/evaluation/__init__.py](../experiments/state_transfer/evaluation/__init__.py) | Python package marker | — |
| [experiments/state_transfer/evaluation/state_analysis.py](../experiments/state_transfer/evaluation/state_analysis.py) | Natural object-state retrieval: five held-out-noun folds and paired inference. | `experiments/state_transfer/training/bridge.py`<br>`shared/data.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/state_transfer/evaluation/state_identity_followup.py](../experiments/state_transfer/evaluation/state_identity_followup.py) | Exploratory, post-primary object-preservation follow-up. Main results unchanged. | `experiments/state_transfer/evaluation/state_analysis.py`<br>`shared/data.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/training.py` |
| [experiments/state_transfer/reporting/__init__.py](../experiments/state_transfer/reporting/__init__.py) | Python package marker | — |
| [experiments/state_transfer/reporting/paper_pdf.py](../experiments/state_transfer/reporting/paper_pdf.py) | Render the working manuscript with ReportLab (bundled PDF runtime). | `shared/paths.py` |
| [experiments/state_transfer/reporting/state_report.py](../experiments/state_transfer/reporting/state_report.py) | Generate figures, a companion report and an archival draft from saved state scores. | `shared/paths.py` |
| [experiments/state_transfer/tests/__init__.py](../experiments/state_transfer/tests/__init__.py) | Python package marker | — |
| [experiments/state_transfer/tests/test_state_identity.py](../experiments/state_transfer/tests/test_state_identity.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/state_transfer/evaluation/state_identity_followup.py`<br>`shared/paths.py` |
| [experiments/state_transfer/tests/test_state_transfer.py](../experiments/state_transfer/tests/test_state_transfer.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/state_transfer/data/state_data.py`<br>`experiments/state_transfer/evaluation/state_analysis.py`<br>`shared/paths.py` |
| [experiments/state_transfer/training/__init__.py](../experiments/state_transfer/training/__init__.py) | Python package marker | — |
| [experiments/state_transfer/training/bridge.py](../experiments/state_transfer/training/bridge.py) | Training: fit the shared state-transfer bridge on paired fitting rows only. | `shared/training.py` |

### multimodal_structure — Exploratory released-label study

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/multimodal_structure/PLAN.md](../experiments/multimodal_structure/PLAN.md) | Completed through week 8: released-label scope | — |
| [experiments/multimodal_structure/README.md](../experiments/multimodal_structure/README.md) | Cross-modal structure experiment | — |
| [experiments/multimodal_structure/__init__.py](../experiments/multimodal_structure/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/configs/models.json](../experiments/multimodal_structure/configs/models.json) | Fixed experiment/model settings | — |
| [experiments/multimodal_structure/configs/pilot.json](../experiments/multimodal_structure/configs/pilot.json) | Fixed experiment/model settings | — |
| [experiments/multimodal_structure/configs/week8.json](../experiments/multimodal_structure/configs/week8.json) | Fixed experiment/model settings | — |
| [experiments/multimodal_structure/data/__init__.py](../experiments/multimodal_structure/data/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/data/audio.py](../experiments/multimodal_structure/data/audio.py) | Load real waveform samples; padding is never counted as observed sound. | — |
| [experiments/multimodal_structure/data/audit.py](../experiments/multimodal_structure/data/audit.py) | Pre-extraction media QA; these checks do not claim human label verification. | `experiments/multimodal_structure/data/audio.py`<br>`experiments/multimodal_structure/data/expanded.py`<br>`experiments/multimodal_structure/data/prepare.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/data/av_data.py](../experiments/multimodal_structure/data/av_data.py) | Turn locally accessible audiovisual clips into separately encoded channels. | — |
| [experiments/multimodal_structure/data/axes.py](../experiments/multimodal_structure/data/axes.py) | Explicit category axes derived from released event labels, not physical states. | — |
| [experiments/multimodal_structure/data/download_vggsounder.py](../experiments/multimodal_structure/data/download_vggsounder.py) | Download a small, pinned, real audiovisual pilot (never the whole dataset). | `experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/data/split.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/data/examples/concepts.csv](../experiments/multimodal_structure/data/examples/concepts.csv) | Dataset acquisition, validation, splitting or manifest preparation | — |
| [experiments/multimodal_structure/data/examples/records.csv](../experiments/multimodal_structure/data/examples/records.csv) | Dataset acquisition, validation, splitting or manifest preparation | — |
| [experiments/multimodal_structure/data/expanded.py](../experiments/multimodal_structure/data/expanded.py) | Plan and download a larger study before any model sees the new test data. | `experiments/multimodal_structure/data/axes.py`<br>`experiments/multimodal_structure/data/download_vggsounder.py`<br>`experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/data/split.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/data/legacy.py](../experiments/multimodal_structure/data/legacy.py) | Import existing state features without running or modifying the old study. | `experiments/multimodal_structure/extraction/cache.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/data/prepare.py](../experiments/multimodal_structure/data/prepare.py) | Validate a simple CSV and freeze it as a portable JSON dataset manifest. | `shared/io.py` |
| [experiments/multimodal_structure/data/setup_week8_models.py](../experiments/multimodal_structure/data/setup_week8_models.py) | Download pinned extra backbones and reuse audited existing pilot model specs. | `experiments/multimodal_structure/data/download_vggsounder.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/data/split.py](../experiments/multimodal_structure/data/split.py) | Assign whole families while linking any families that share a source. | — |
| [experiments/multimodal_structure/docs/DATA_GUIDE.md](../experiments/multimodal_structure/docs/DATA_GUIDE.md) | Data guide | — |
| [experiments/multimodal_structure/docs/METHODS.md](../experiments/multimodal_structure/docs/METHODS.md) | Methods | — |
| [experiments/multimodal_structure/docs/PILOT_SCORES.csv](../experiments/multimodal_structure/docs/PILOT_SCORES.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/PROGRESS_REPORT.md](../experiments/multimodal_structure/docs/PROGRESS_REPORT.md) | Cross-modal representation study: progress report | — |
| [experiments/multimodal_structure/docs/RESULTS.md](../experiments/multimodal_structure/docs/RESULTS.md) | Actual execution results | — |
| [experiments/multimodal_structure/docs/WEEK8_PROTOCOL.md](../experiments/multimodal_structure/docs/WEEK8_PROTOCOL.md) | Locked held-out study: October 23, 2026 | — |
| [experiments/multimodal_structure/docs/WEEK8_RUN_GUIDE.md](../experiments/multimodal_structure/docs/WEEK8_RUN_GUIDE.md) | Reproducing the expanded study | — |
| [experiments/multimodal_structure/docs/run_summary.json](../experiments/multimodal_structure/docs/run_summary.json) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/CAV_CHANCE_COMPARISONS.csv](../experiments/multimodal_structure/docs/week8/CAV_CHANCE_COMPARISONS.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/CHANCE_BASELINES.csv](../experiments/multimodal_structure/docs/week8/CHANCE_BASELINES.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/RESULTS.md](../experiments/multimodal_structure/docs/week8/RESULTS.md) | Week 8: locked held-out released-label results | — |
| [experiments/multimodal_structure/docs/week8/RESULTS_EXPLAINED.md](../experiments/multimodal_structure/docs/week8/RESULTS_EXPLAINED.md) | Results explained: chance baselines and worked calculations | — |
| [experiments/multimodal_structure/docs/week8/VALIDATION.md](../experiments/multimodal_structure/docs/week8/VALIDATION.md) | Validation and interpretation notes | — |
| [experiments/multimodal_structure/docs/week8/category_scores.csv](../experiments/multimodal_structure/docs/week8/category_scores.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/category_transfer.png](../experiments/multimodal_structure/docs/week8/category_transfer.png) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/chance_distributions.png](../experiments/multimodal_structure/docs/week8/chance_distributions.png) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/concepts_and_axes.csv](../experiments/multimodal_structure/docs/week8/concepts_and_axes.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/geometry_heatmaps.png](../experiments/multimodal_structure/docs/week8/geometry_heatmaps.png) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/geometry_scores.csv](../experiments/multimodal_structure/docs/week8/geometry_scores.csv) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/docs/week8/run_summary.json](../experiments/multimodal_structure/docs/week8/run_summary.json) | Protocol, interpretation, planning or compact result export | — |
| [experiments/multimodal_structure/evaluation/__init__.py](../experiments/multimodal_structure/evaluation/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/evaluation/cka.py](../experiments/multimodal_structure/evaluation/cka.py) | Linear CKA compares centered geometry, even when feature widths differ. | — |
| [experiments/multimodal_structure/evaluation/controls.py](../experiments/multimodal_structure/evaluation/controls.py) | Correspondence shuffles preserve each model's entire internal geometry. | `experiments/multimodal_structure/evaluation/cka.py`<br>`experiments/multimodal_structure/evaluation/neighbors.py`<br>`experiments/multimodal_structure/evaluation/rsa.py` |
| [experiments/multimodal_structure/evaluation/directions.py](../experiments/multimodal_structure/evaluation/directions.py) | Evaluate a trained CAV without treating it as a centroid displacement. | `experiments/multimodal_structure/training/alignment.py`<br>`experiments/multimodal_structure/training/probes.py` |
| [experiments/multimodal_structure/evaluation/evaluate.py](../experiments/multimodal_structure/evaluation/evaluate.py) | Compare all supplied model caches on exactly the same held-out concepts. | `experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/evaluation/controls.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/evaluation/fit_directions.py](../experiments/multimodal_structure/evaluation/fit_directions.py) | Fit train-only maps/probes and evaluate each relation on untouched families. | `experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/evaluation/directions.py`<br>`experiments/multimodal_structure/evaluation/evaluate.py`<br>`experiments/multimodal_structure/evaluation/retrieval.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`experiments/multimodal_structure/training/alignment.py`<br>`experiments/multimodal_structure/training/preprocessing.py`<br>`experiments/multimodal_structure/training/probes.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/evaluation/locked.py](../experiments/multimodal_structure/evaluation/locked.py) | Week 8: score saved fits once, after locking settings and software identity. | `experiments/multimodal_structure/data/axes.py`<br>`experiments/multimodal_structure/evaluation/directions.py`<br>`experiments/multimodal_structure/evaluation/evaluate.py`<br>`experiments/multimodal_structure/evaluation/neighbors.py`<br>`experiments/multimodal_structure/evaluation/rsa.py`<br>`experiments/multimodal_structure/provenance.py`<br>`experiments/multimodal_structure/training/alignment.py`<br>`experiments/multimodal_structure/training/build_fits.py`<br>`experiments/multimodal_structure/training/preprocessing.py`<br>`experiments/multimodal_structure/training/probes.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/evaluation/neighbors.py](../experiments/multimodal_structure/evaluation/neighbors.py) | Labeled neighborhood agreement, rather than unlabeled graph isomorphism. | `experiments/multimodal_structure/evaluation/rsa.py` |
| [experiments/multimodal_structure/evaluation/reliability.py](../experiments/multimodal_structure/evaluation/reliability.py) | Development-only repeatability, estimated without touching test scores. | `experiments/multimodal_structure/evaluation/cka.py`<br>`experiments/multimodal_structure/evaluation/rsa.py`<br>`experiments/multimodal_structure/training/build_fits.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/evaluation/retrieval.py](../experiments/multimodal_structure/evaluation/retrieval.py) | Relation retrieval retains object/event identity as an explicit guardrail. | `experiments/multimodal_structure/evaluation/rsa.py` |
| [experiments/multimodal_structure/evaluation/rsa.py](../experiments/multimodal_structure/evaluation/rsa.py) | Compare concept distance orderings without aligning feature coordinates. | — |
| [experiments/multimodal_structure/extraction/__init__.py](../experiments/multimodal_structure/extraction/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/extraction/audio.py](../experiments/multimodal_structure/extraction/audio.py) | BEATs pretraining features from the official local implementation. | `experiments/multimodal_structure/data/audio.py` |
| [experiments/multimodal_structure/extraction/audiomae.py](../experiments/multimodal_structure/extraction/audiomae.py) | Audio-only AudioMAE pretraining backbone, with its published processor. | `experiments/multimodal_structure/data/audio.py` |
| [experiments/multimodal_structure/extraction/cache.py](../experiments/multimodal_structure/extraction/cache.py) | Explicit cache identity and atomic writes prevent silent model/row mixups. | `shared/io.py` |
| [experiments/multimodal_structure/extraction/extract.py](../experiments/multimodal_structure/extraction/extract.py) | Extract one model into checked, resumable batches, then assemble one cache. | `experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/extraction/__init__.py`<br>`experiments/multimodal_structure/extraction/audio.py`<br>`experiments/multimodal_structure/extraction/audiomae.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`experiments/multimodal_structure/extraction/hf_vision.py`<br>`experiments/multimodal_structure/extraction/text.py`<br>`experiments/multimodal_structure/extraction/vision.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/extraction/hf_vision.py](../experiments/multimodal_structure/extraction/hf_vision.py) | Original image-only MAE backbone through the existing Transformers runtime. | — |
| [experiments/multimodal_structure/extraction/text.py](../experiments/multimodal_structure/extraction/text.py) | Frozen text-only transformer extraction with masked mean pooling. | — |
| [experiments/multimodal_structure/extraction/vision.py](../experiments/multimodal_structure/extraction/vision.py) | Frozen timm vision transformer extraction, reusing the baseline convention. | — |
| [experiments/multimodal_structure/provenance.py](../experiments/multimodal_structure/provenance.py) | Small byte-identity helpers shared by fitting and locked evaluation. | — |
| [experiments/multimodal_structure/reporting/__init__.py](../experiments/multimodal_structure/reporting/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/reporting/report.py](../experiments/multimodal_structure/reporting/report.py) | Readable report from saved scores; the report never reruns model fitting. | — |
| [experiments/multimodal_structure/reporting/week8.py](../experiments/multimodal_structure/reporting/week8.py) | Export the locked week-8 results, including compact tables and static plots. | `experiments/multimodal_structure/data/axes.py` |
| [experiments/multimodal_structure/requirements.txt](../experiments/multimodal_structure/requirements.txt) | Experiment-specific dependencies | — |
| [experiments/multimodal_structure/run.py](../experiments/multimodal_structure/run.py) | Run as: python -m experiments.multimodal_structure.run --help (from sz684). | `experiments/multimodal_structure/data/av_data.py`<br>`experiments/multimodal_structure/data/download_vggsounder.py`<br>`experiments/multimodal_structure/data/expanded.py`<br>`experiments/multimodal_structure/data/legacy.py`<br>`experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/data/setup_week8_models.py`<br>`experiments/multimodal_structure/evaluation/evaluate.py`<br>`experiments/multimodal_structure/evaluation/fit_directions.py`<br>`experiments/multimodal_structure/evaluation/locked.py`<br>`experiments/multimodal_structure/extraction/extract.py`<br>`experiments/multimodal_structure/reporting/report.py`<br>`experiments/multimodal_structure/reporting/week8.py`<br>`experiments/multimodal_structure/training/build_fits.py` |
| [experiments/multimodal_structure/tests/__init__.py](../experiments/multimodal_structure/tests/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/tests/test_audio.py](../experiments/multimodal_structure/tests/test_audio.py) | Real PCM WAV decoding/resampling checks, without model downloads. | `experiments/multimodal_structure/data/audio.py` |
| [experiments/multimodal_structure/tests/test_locked_study.py](../experiments/multimodal_structure/tests/test_locked_study.py) | The fit/lock boundary must prevent test-driven parameter changes. | `experiments/multimodal_structure/data/expanded.py`<br>`experiments/multimodal_structure/evaluation/locked.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`experiments/multimodal_structure/training/build_fits.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/tests/test_metrics.py](../experiments/multimodal_structure/tests/test_metrics.py) | Known transformations check scientific meanings, not implementation details. | `experiments/multimodal_structure/evaluation/cka.py`<br>`experiments/multimodal_structure/evaluation/directions.py`<br>`experiments/multimodal_structure/evaluation/neighbors.py`<br>`experiments/multimodal_structure/evaluation/retrieval.py`<br>`experiments/multimodal_structure/evaluation/rsa.py`<br>`experiments/multimodal_structure/training/alignment.py`<br>`experiments/multimodal_structure/training/probes.py` |
| [experiments/multimodal_structure/tests/test_pipeline.py](../experiments/multimodal_structure/tests/test_pipeline.py) | A tiny synthetic three-modality fixture verifies orchestration, not science. | `experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/data/split.py`<br>`experiments/multimodal_structure/evaluation/evaluate.py`<br>`experiments/multimodal_structure/evaluation/fit_directions.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`experiments/multimodal_structure/reporting/report.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/training/__init__.py](../experiments/multimodal_structure/training/__init__.py) | Python package marker | — |
| [experiments/multimodal_structure/training/alignment.py](../experiments/multimodal_structure/training/alignment.py) | Orthogonal maps use row vectors throughout: target approximately source @ R. | — |
| [experiments/multimodal_structure/training/build_fits.py](../experiments/multimodal_structure/training/build_fits.py) | Week 7: fit on training sources, select settings on validation, never score test. | `experiments/multimodal_structure/data/axes.py`<br>`experiments/multimodal_structure/data/prepare.py`<br>`experiments/multimodal_structure/evaluation/directions.py`<br>`experiments/multimodal_structure/evaluation/evaluate.py`<br>`experiments/multimodal_structure/evaluation/fit_directions.py`<br>`experiments/multimodal_structure/extraction/cache.py`<br>`experiments/multimodal_structure/provenance.py`<br>`experiments/multimodal_structure/training/alignment.py`<br>`experiments/multimodal_structure/training/preprocessing.py`<br>`experiments/multimodal_structure/training/probes.py`<br>`shared/io.py` |
| [experiments/multimodal_structure/training/preprocessing.py](../experiments/multimodal_structure/training/preprocessing.py) | Small serializable train-only transforms for alignment and probes. | — |
| [experiments/multimodal_structure/training/probes.py](../experiments/multimodal_structure/training/probes.py) | A ridge binary classifier gives a simple, regularized CAV baseline. | — |

### baseline — Historical; active reproduction dependency

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/baseline/README.md](../experiments/baseline/README.md) | Historical: WIT alignment baseline | — |
| [experiments/baseline/__init__.py](../experiments/baseline/__init__.py) | Python package marker | — |
| [experiments/baseline/configs/mac.json](../experiments/baseline/configs/mac.json) | Fixed experiment/model settings | — |
| [experiments/baseline/data/__init__.py](../experiments/baseline/data/__init__.py) | Python package marker | — |
| [experiments/baseline/data/experiment.py](../experiments/baseline/data/experiment.py) | Mac-sized PRH replication. See README.md for protocol and limitations. | `experiments/baseline/evaluation/alignment.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/baseline/docs/LEGACY_BASELINE.md](../experiments/baseline/docs/LEGACY_BASELINE.md) | Mac-scale Platonic Representation Hypothesis replication | — |
| [experiments/baseline/docs/PROTOCOL.md](../experiments/baseline/docs/PROTOCOL.md) | Frozen reduced experiment protocol | — |
| [experiments/baseline/evaluation/__init__.py](../experiments/baseline/evaluation/__init__.py) | Python package marker | — |
| [experiments/baseline/evaluation/alignment.py](../experiments/baseline/evaluation/alignment.py) | CPU float32 reference for the authors' mutual-neighbor measurement. | `shared/paths.py` |
| [experiments/baseline/evaluation/language_eval.py](../experiments/baseline/evaluation/language_eval.py) | Small, explicitly non-paper OpenWebText evaluation for the Mac baseline. | `shared/io.py`<br>`shared/paths.py` |
| [experiments/baseline/evaluation/verify_extraction.py](../experiments/baseline/evaluation/verify_extraction.py) | Compare saved pilot features with upstream-style extraction on the first pair. | `experiments/baseline/data/experiment.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/baseline/reporting/__init__.py](../experiments/baseline/reporting/__init__.py) | Python package marker | — |
| [experiments/baseline/reporting/report.py](../experiments/baseline/reporting/report.py) | Build a descriptive reduced-protocol report, never a paper-replication verdict. | `shared/io.py`<br>`shared/paths.py` |
| [experiments/baseline/tests/__init__.py](../experiments/baseline/tests/__init__.py) | Python package marker | — |
| [experiments/baseline/tests/test_alignment.py](../experiments/baseline/tests/test_alignment.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/baseline/evaluation/alignment.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/baseline/tests/test_cache.py](../experiments/baseline/tests/test_cache.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/baseline/data/experiment.py`<br>`shared/paths.py` |
| [experiments/baseline/tests/test_language_eval.py](../experiments/baseline/tests/test_language_eval.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/baseline/evaluation/language_eval.py`<br>`shared/paths.py` |

### heldout_layers — Historical

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/heldout_layers/README.md](../experiments/heldout_layers/README.md) | Historical: held-out layer selection | — |
| [experiments/heldout_layers/__init__.py](../experiments/heldout_layers/__init__.py) | Python package marker | — |
| [experiments/heldout_layers/docs/RESEARCH_PLAN.md](../experiments/heldout_layers/docs/RESEARCH_PLAN.md) | Work-session research plan | — |
| [experiments/heldout_layers/evaluation/__init__.py](../experiments/heldout_layers/evaluation/__init__.py) | Python package marker | — |
| [experiments/heldout_layers/evaluation/heldout_layers.py](../experiments/heldout_layers/evaluation/heldout_layers.py) | Prespecified held-out layer-selection sensitivity study; see RESEARCH_PLAN.md. | `experiments/baseline/data/experiment.py`<br>`experiments/baseline/evaluation/alignment.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/heldout_layers/tests/__init__.py](../experiments/heldout_layers/tests/__init__.py) | Python package marker | — |
| [experiments/heldout_layers/tests/test_heldout_layers.py](../experiments/heldout_layers/tests/test_heldout_layers.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/baseline/evaluation/alignment.py`<br>`experiments/heldout_layers/evaluation/heldout_layers.py`<br>`shared/paths.py` |

### robustness — Historical

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/robustness/README.md](../experiments/robustness/README.md) | Historical: alignment robustness | — |
| [experiments/robustness/__init__.py](../experiments/robustness/__init__.py) | Python package marker | — |
| [experiments/robustness/docs/ROBUSTNESS_PROTOCOL.md](../experiments/robustness/docs/ROBUSTNESS_PROTOCOL.md) | Sample-size and preprocessing sensitivity: frozen protocol | — |
| [experiments/robustness/evaluation/__init__.py](../experiments/robustness/evaluation/__init__.py) | Python package marker | — |
| [experiments/robustness/evaluation/robustness.py](../experiments/robustness/evaluation/robustness.py) | Run ROBUSTNESS_PROTOCOL.md using cached features only. | `experiments/baseline/data/experiment.py`<br>`experiments/baseline/evaluation/alignment.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/transforms.py` |
| [experiments/robustness/tests/__init__.py](../experiments/robustness/tests/__init__.py) | Python package marker | — |
| [experiments/robustness/tests/test_robustness.py](../experiments/robustness/tests/test_robustness.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/baseline/evaluation/alignment.py`<br>`experiments/robustness/evaluation/robustness.py`<br>`shared/paths.py`<br>`shared/transforms.py` |

### paired_uncertainty — Historical

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/paired_uncertainty/README.md](../experiments/paired_uncertainty/README.md) | Historical: paired alignment uncertainty | — |
| [experiments/paired_uncertainty/__init__.py](../experiments/paired_uncertainty/__init__.py) | Python package marker | — |
| [experiments/paired_uncertainty/docs/PAIRED_PROTOCOL.md](../experiments/paired_uncertainty/docs/PAIRED_PROTOCOL.md) | Paired difference uncertainty: protocol fixed before this analysis | — |
| [experiments/paired_uncertainty/evaluation/__init__.py](../experiments/paired_uncertainty/evaluation/__init__.py) | Python package marker | — |
| [experiments/paired_uncertainty/evaluation/paired_uncertainty.py](../experiments/paired_uncertainty/evaluation/paired_uncertainty.py) | Conditional paired query bootstrap; see PAIRED_PROTOCOL.md. | `experiments/baseline/data/experiment.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/transforms.py` |
| [experiments/paired_uncertainty/tests/__init__.py](../experiments/paired_uncertainty/tests/__init__.py) | Python package marker | — |
| [experiments/paired_uncertainty/tests/test_paired_uncertainty.py](../experiments/paired_uncertainty/tests/test_paired_uncertainty.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/paired_uncertainty/evaluation/paired_uncertainty.py`<br>`shared/paths.py` |

### difference_transfer — Historical

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/difference_transfer/README.md](../experiments/difference_transfer/README.md) | Historical: difference-vector transfer | — |
| [experiments/difference_transfer/__init__.py](../experiments/difference_transfer/__init__.py) | Python package marker | — |
| [experiments/difference_transfer/docs/TRANSFER_PROTOCOL.md](../experiments/difference_transfer/docs/TRANSFER_PROTOCOL.md) | Cross-modal difference-vector transfer pilot | — |
| [experiments/difference_transfer/evaluation/__init__.py](../experiments/difference_transfer/evaluation/__init__.py) | Python package marker | — |
| [experiments/difference_transfer/evaluation/difference_transfer.py](../experiments/difference_transfer/evaluation/difference_transfer.py) | Supervised held-out pair-difference pilot; not a semantic relation benchmark. | `experiments/baseline/data/experiment.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/training.py`<br>`shared/transforms.py` |
| [experiments/difference_transfer/tests/__init__.py](../experiments/difference_transfer/tests/__init__.py) | Python package marker | — |
| [experiments/difference_transfer/tests/test_difference_transfer.py](../experiments/difference_transfer/tests/test_difference_transfer.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/difference_transfer/evaluation/difference_transfer.py`<br>`shared/paths.py`<br>`shared/training.py` |

### endpoint_audit — Historical

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/endpoint_audit/README.md](../experiments/endpoint_audit/README.md) | Historical: endpoint contribution audit | — |
| [experiments/endpoint_audit/__init__.py](../experiments/endpoint_audit/__init__.py) | Python package marker | — |
| [experiments/endpoint_audit/docs/ENDPOINT_AUDIT_PROTOCOL.md](../experiments/endpoint_audit/docs/ENDPOINT_AUDIT_PROTOCOL.md) | Endpoint audit — October 15, 2026 | — |
| [experiments/endpoint_audit/evaluation/__init__.py](../experiments/endpoint_audit/evaluation/__init__.py) | Python package marker | — |
| [experiments/endpoint_audit/evaluation/endpoint_audit.py](../experiments/endpoint_audit/evaluation/endpoint_audit.py) | Diagnose endpoint contributions; export a self-contained vector explorer. | `experiments/baseline/data/experiment.py`<br>`experiments/difference_transfer/evaluation/difference_transfer.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/training.py`<br>`shared/transforms.py` |
| [experiments/endpoint_audit/reporting/vector_explorer.html](../experiments/endpoint_audit/reporting/vector_explorer.html) | Saved-result reporting, rendering or interactive explorer | — |
| [experiments/endpoint_audit/tests/__init__.py](../experiments/endpoint_audit/tests/__init__.py) | Python package marker | — |
| [experiments/endpoint_audit/tests/test_endpoint_audit.py](../experiments/endpoint_audit/tests/test_endpoint_audit.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/endpoint_audit/evaluation/endpoint_audit.py`<br>`shared/paths.py` |

### color_transfer — Retired; partial artifacts preserved

| File | Purpose | Local imports |
| --- | --- | --- |
| [experiments/color_transfer/README.md](../experiments/color_transfer/README.md) | Retired: red → blue sketch transfer | — |
| [experiments/color_transfer/__init__.py](../experiments/color_transfer/__init__.py) | Python package marker | — |
| [experiments/color_transfer/configs/semantic_quickdraw.json](../experiments/color_transfer/configs/semantic_quickdraw.json) | Fixed experiment/model settings | — |
| [experiments/color_transfer/data/__init__.py](../experiments/color_transfer/data/__init__.py) | Python package marker | — |
| [experiments/color_transfer/data/quickdraw_data.py](../experiments/color_transfer/data/quickdraw_data.py) | Download a bounded Quick, Draw! subset and render exact RGB interventions. | `shared/io.py`<br>`shared/paths.py` |
| [experiments/color_transfer/data/semantic_extract.py](../experiments/color_transfer/data/semantic_extract.py) | Pinned offline model extraction, unique text prompts, resumable vision batches. | `shared/cache.py`<br>`shared/io.py`<br>`shared/paths.py` |
| [experiments/color_transfer/docs/CONCEPT_TRANSFER_NEXT.md](../experiments/color_transfer/docs/CONCEPT_TRANSFER_NEXT.md) | Next experiment: repeated semantic relations across modalities | — |
| [experiments/color_transfer/docs/PAPER_DRAFT.md](../experiments/color_transfer/docs/PAPER_DRAFT.md) | From Endpoint Alignment to Relation Transfer: Evaluating Cross-Modal Vector Arithmetic | — |
| [experiments/color_transfer/docs/QUICKDRAW_PROTOCOL.md](../experiments/color_transfer/docs/QUICKDRAW_PROTOCOL.md) | Executable controlled sketch protocol | — |
| [experiments/color_transfer/docs/SEMANTIC_PROTOCOL.md](../experiments/color_transfer/docs/SEMANTIC_PROTOCOL.md) | Proposed confirmatory experiment: reusable red→blue transfer | — |
| [experiments/color_transfer/evaluation/__init__.py](../experiments/color_transfer/evaluation/__init__.py) | Python package marker | — |
| [experiments/color_transfer/evaluation/pilot_planning.py](../experiments/color_transfer/evaluation/pilot_planning.py) | Rough sensitivity planning from four retired pilot rotations, not a power guarantee. | `shared/io.py`<br>`shared/paths.py` |
| [experiments/color_transfer/evaluation/semantic_analysis.py](../experiments/color_transfer/evaluation/semantic_analysis.py) | Retired RGB scoring pipeline; reusable helpers now live in shared/. | `shared/data.py`<br>`shared/io.py`<br>`shared/paths.py`<br>`shared/training.py` |
| [experiments/color_transfer/tests/__init__.py](../experiments/color_transfer/tests/__init__.py) | Python package marker | — |
| [experiments/color_transfer/tests/test_semantic.py](../experiments/color_transfer/tests/test_semantic.py) | Software behavior check; synthetic fixtures are not empirical findings | `experiments/color_transfer/data/quickdraw_data.py`<br>`experiments/color_transfer/evaluation/semantic_analysis.py`<br>`shared/data.py`<br>`shared/paths.py` |
| [experiments/color_transfer/training/README.md](../experiments/color_transfer/training/README.md) | Training | — |
| [experiments/color_transfer/training/__init__.py](../experiments/color_transfer/training/__init__.py) | Python package marker | — |

## Runtime file dependencies and deletion decisions

- `state_transfer/data/state_data.py` reads `baseline/results/baseline/manifest.json` for pinned encoder revisions. Keep that manifest even if the WIT experiment is not being developed.
- Historical held-out, robustness, paired-uncertainty, difference and endpoint analyses read baseline features/results; difference and endpoint work also read held-out-layer selections. The endpoint audit reads difference-transfer artifacts.
- `state_transfer/evaluation/state_identity_followup.py` imports scoring helpers from `state_analysis.py` and writes its own `secondary_identity/` results. This is an existing exploratory follow-up, not Track 2.
- Feature extraction reads manifests, image/text inputs, pinned weights and cache identities. `shared/extractor_migrations.json` must stay paired with unmodified historical extractors.
- `shared/paths.py` resolves legacy names embedded in historical artifacts. Saved snapshots retain their original paths and hashes; do not rewrite those records for cosmetic consistency.
- Multimodal extraction reads local pinned model specifications and downloaded assets under its results tree; fits and evaluation locks bind feature/data/source identities.
- Both reporting pipelines consume saved scores. State PDF generation also reads the editorial manuscripts and generated figures. Compact multimodal CSV/JSON/PNG exports under `docs/` remain in their original locations.
- Retired color source, partial caches and protocols are kept because incomplete extraction is part of the historical record. No substantive research file is established as safely redundant by age alone.

Coverage at this snapshot: **192 individually listed maintained files + 43,575 recursively grouped local files = 43,767 files under sz684/**. Empty directories contain no files and are not counted. No Track 1/2 implementation package exists.
