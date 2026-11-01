# Retired: red → blue sketch transfer

Quick, Draw! color sanity check. Main extraction was interrupted before test scoring. Preserve artifacts; do not resume without a new instruction.

| Folder | Contents |
| --- | --- |
| [data/](data/) | Data acquisition, manifests, frozen feature extraction |
| [training/](training/) | Translator fitting (frozen encoders are not trained) |
| [evaluation/](evaluation/) | Experiment scoring, controls, validation, and inference |
| [configs/](configs/) | Experiment parameters |
| [docs/](docs/) | Protocols, manuscripts, and study notes |
| [tests/](tests/) | Experiment-specific verification |
| [results/](results/) | Saved run data, features, metrics, and frozen source snapshots (Git-ignored) |

The ridge solver lives in [shared/training.py](../../shared/training.py); train-fitted transformations live in [shared/](../../shared/). Evaluation scripts fit the map before scoring.
