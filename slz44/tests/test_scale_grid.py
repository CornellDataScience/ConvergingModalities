import json
from argparse import Namespace
import numpy as np
from prepare_data import heldout
from scale_grid import main, scaled_k
from test_pipeline import make_run


def test_scaled_k_matches_protocol():
    assert [scaled_k(n, 512) for n in (128, 256, 512)] == [2, 5, 10]


def test_scale_grid_end_to_end(tmp_path):
    run = tmp_path / 'baseline'
    make_run(run)
    held = tmp_path / 'heldout' / 'results.json'
    heldout(Namespace(baseline=run, out=held, threads=1))
    out = tmp_path / 'out'
    main(['--baseline', str(run), '--heldout', str(held), '--out', str(out), '--sizes', '16', '32', '--permutations', '10'])
    rows = json.loads((out / 'results.json').read_text())['rows']
    assert len(rows) == 18 * 6 * 2
    assert all(np.isfinite(r['cka']) and np.isfinite(r['gw']) for r in rows)
    assert (out / 'REPORT.md').exists() and (out / 'scale_grid.png').exists()
