# Cross-modal representation experiments

The latest experiment tests whether a change learned in words, such as **raw → cooked**, guides image retrieval for object categories excluded from training.

Start with the [experiment index and commands](sz684/README.md). Each study owns its data preparation, training/evaluation code, documentation, tests, and saved results.

- [Natural object-state transfer — latest study](sz684/experiments/state_transfer/)
- [Difference-vector transfer](sz684/experiments/difference_transfer/)
- [Endpoint audit](sz684/experiments/endpoint_audit/)
- [WIT baseline](sz684/experiments/baseline/)
- [Held-out layer selection](sz684/experiments/heldout_layers/)
- [Robustness](sz684/experiments/robustness/)
- [Paired uncertainty](sz684/experiments/paired_uncertainty/)
- [Color transfer — retired](sz684/experiments/color_transfer/)

Reusable transformations and the ridge solver live in [shared/](sz684/shared/). Run tests from this directory with `sz684/.venv/bin/python -m pytest -q`.
