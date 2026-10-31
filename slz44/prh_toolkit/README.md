# prh-toolkit (draft)

Shared tools for testing representational convergence between models: anisotropy controls, alignment metrics,
null models, state-direction and retrieval tools, and one results format that a visualizer can read.

    pip install -e ".[dev]"
    pytest
    prh validate results/*.json

Status: the results format (`prh.schema`) and the `prh validate` command exist. The other modules are placeholders
for each author's tested code. See `docs/RESULTS_SCHEMA.md` and `docs/CONTRIBUTING.md`.

## Visualizer

Open `visualizer/index.html` in a browser (no install, no server). It starts with sample data; drop your own
results files onto the page to replace it. Convert existing outputs first:

    python tools/convert_slz44.py scale-grid ../results/scale_grid/results.json ../SCALE_GRID_PROTOCOL.md --out out/scale_grid.json
    python tools/convert_slz44.py state-matrix ../results/state_directions/results.json ../STATE_DIRECTIONS_PROTOCOL.md --out out/state_matrix.json
    python tools/convert_slz44.py categories ../results/state_directions/categories.csv ../STATE_DIRECTIONS_PROTOCOL.md --out out/categories.json

Views: metric grid (metric by condition by N), alignment (matrix and per-relation match), and placeholders for
Track 2's strength-versus-retention curve and example gallery.
