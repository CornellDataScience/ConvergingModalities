import json
import numpy as np
from anisotropy import digest
from metric_controls import main
from prepare_data import heldout
from argparse import Namespace
from test_pipeline import make_run


def test_metric_controls_end_to_end(tmp_path):
    run = tmp_path / 'baseline'
    make_run(run)
    held = tmp_path / 'heldout' / 'results.json'
    heldout(Namespace(baseline=run, out=held, threads=1))
    out = tmp_path / 'out'
    main(['--baseline', str(run), '--heldout', str(held), '--out', str(out), '--permutations', '20'])
    rows = json.loads((out / 'results.json').read_text())['rows']
    assert len(rows) == 18 * 6
    assert (out / 'REPORT.md').exists()
    assert all(np.isfinite(r['cka_score']) and np.isfinite(r['gw_reduction']) for r in rows)
