# Completed reference study: natural object-state transfer

**Status:** completed reference study; identity follow-ups are exploratory. The proposed two-track extension is [planning only](../../docs/planning/STATE_RELATIONS_PLAN.md).

Cleaning, cooking (raw → cooked), ripening, thawing, and drying across noun labels excluded from fitting.

| Folder | Contents |
| --- | --- |
| [data/](data/) | Data acquisition, manifests, frozen feature extraction |
| [training/](training/) | Translator fitting (frozen encoders are not trained) |
| [evaluation/](evaluation/) | Experiment scoring, controls, validation, and inference |
| [reporting/](reporting/) | Reports, figures, manuscripts/PDF rendering |
| [configs/](configs/) | Experiment parameters |
| [docs/](docs/) | Protocols, manuscripts, and study notes |
| [tests/](tests/) | Experiment-specific verification |
| [results/](results/) | Saved run data, features, metrics, and frozen source snapshots (Git-ignored) |
| [output/](output/) | Rendered figures and PDFs |

Run from `sz684/`:

```sh
.venv/bin/python run.py state-evaluate
```

Read [the accessible paper](docs/STATE_TRANSFER_PAPER_V2.md) or [the primary protocol](docs/STATE_TRANSFER_PROTOCOL.md). The supervised translator is fitted in [training/bridge.py](training/bridge.py), called once per fold by [evaluation/state_analysis.py](evaluation/state_analysis.py). Feature extraction belongs to [data/state_extract.py](data/state_extract.py).

```sh
.venv/bin/python run.py state-data
.venv/bin/python run.py state-extract --modality language --device mps
.venv/bin/python run.py state-extract --modality vision --device mps
.venv/bin/python run.py state-evaluate
.venv/bin/python run.py state-followup
.venv/bin/python run.py state-report
```

The pipeline uses pinned revisions from the earlier baseline manifest and locally cached weights. Run modalities sequentially. Analysis/reporting regenerate outputs; preserve a copy of historical results before intentionally rerunning.

First-result state-and-noun success is 38.7%, versus 20.9% without a state instruction and 42.1% for direct target text. The primary AP gain is positive, but noun retention falls from 86.6% to 79.4%, failing the preservation criterion. These are independent photographs with weak released labels, not paired physical transitions.

`results/` keeps the original manifest, images, feature caches, metrics, validation curves, bootstrap draws, and frozen `source_snapshot/`. `output/` contains presentation figures/PDFs. The original lock and snapshots retain their pre-reorganization paths/hashes; working-source provenance for new runs includes the separated bridge and shared helpers.
