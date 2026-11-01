# Historical: WIT alignment baseline

Reduced BLOOMZ/DINOv2 replication, feature extraction, and language evaluation.

| Folder | Contents |
| --- | --- |
| [data/](data/) | Data acquisition, manifests, frozen feature extraction |
| [evaluation/](evaluation/) | Experiment scoring, controls, validation, and inference |
| [reporting/](reporting/) | Reports, figures, manuscripts/PDF rendering |
| [configs/](configs/) | Experiment parameters |
| [docs/](docs/) | Protocols, manuscripts, and study notes |
| [tests/](tests/) | Experiment-specific verification |
| [results/](results/) | Saved run data, features, metrics, and frozen source snapshots (Git-ignored) |

Run from `sz684/`:

```sh
.venv/bin/python run.py baseline init --run experiments/baseline/results/pilot --samples 32
```

`data/experiment.py` is the original combined init/extract/align CLI. `init` prepares data, `extract` computes frozen features, and `align` scores them; it does not train encoder weights. Its pilot and baseline runs are separate subdirectories of `results/`.
