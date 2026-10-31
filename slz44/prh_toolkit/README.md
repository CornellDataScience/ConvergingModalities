# prh-toolkit (draft)

Shared tools for testing representational convergence between models: anisotropy controls, alignment metrics,
null models, state-direction and retrieval tools, and one results format that a visualizer can read.

    pip install -e ".[dev]"
    pytest
    prh validate results/*.json

Status: the results format (`prh.schema`) and the `prh validate` command exist. The other modules are placeholders
for each author's tested code. See `docs/RESULTS_SCHEMA.md` and `docs/CONTRIBUTING.md`.
