# Historical: difference-vector transfer

Supervised mapping of arbitrary endpoint differences; this is not the named-state relation experiment.

| Folder | Contents |
| --- | --- |
| [evaluation/](evaluation/) | Experiment scoring, controls, validation, and inference |
| [docs/](docs/) | Protocols, manuscripts, and study notes |
| [tests/](tests/) | Experiment-specific verification |
| [results/](results/) | Saved run data, features, metrics, and frozen source snapshots (Git-ignored) |

Run from `sz684/`:

```sh
.venv/bin/python run.py difference-vectors
```

The ridge solver lives in [shared/training.py](../../shared/training.py); train-fitted transformations live in [shared/](../../shared/). Evaluation scripts fit the map before scoring.
