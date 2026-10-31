"""Check the package against the original experiment scripts in the parent folder (skipped when unavailable).

The scripts `extra_metrics.py` (numpy) and `anisotropy.py` (torch) live next to prh_toolkit. These tests make
sure the ported numpy code gives the same numbers as the code that produced the reported results.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from prh.controls import MODES, apply_control, diagnostics, fit_control, l2_normalize
from prh.metrics import linear_cka, normalized_distance
from prh.nulls import cka_null, gw_cost_null, knn_null

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def features(seed, n=200, d=40):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    x[:, 0] *= 15
    return l2_normalize(x + 3.)


def test_cka_and_gw_match_original_extra_metrics():
    extra = pytest.importorskip('extra_metrics')
    a, b = features(0), features(1)
    assert linear_cka(a, b) == pytest.approx(extra.linear_cka(a, b))
    ours = cka_null(a, b, 30, seed=4)
    theirs = extra.cka_with_null(a, b, 30, 4)
    assert ours['null'] == pytest.approx(theirs['null']) and ours['excess'] == pytest.approx(theirs['excess'])
    ours = gw_cost_null(a, b, 30, seed=4)
    theirs = extra.gw_cost_with_null(normalized_distance(a), normalized_distance(b), 30, 4)
    assert ours['reduction'] == pytest.approx(theirs['reduction'])


def test_controls_match_original_torch_implementation():
    torch = pytest.importorskip('torch')
    original = pytest.importorskip('anisotropy')
    x = features(2)
    train, test = x[:120], x[120:]
    for mode in MODES:
        ours = apply_control(test, fit_control(train, mode))
        theirs = original.apply_control(torch.from_numpy(test).float(),
                                        original.fit_control(torch.from_numpy(train).float(), mode)).numpy()
        # components can differ in sign; compare inner products, which every metric here depends on
        assert np.allclose(ours @ ours.T, theirs @ theirs.T, atol=1e-3), mode
    ours = diagnostics(test)
    theirs = original.diagnostics(torch.from_numpy(test).float())
    for key, value in theirs.items():
        assert ours[key] == pytest.approx(value, rel=1e-3, abs=1e-4)


def test_knn_null_matches_original_measure():
    torch = pytest.importorskip('torch')
    original = pytest.importorskip('anisotropy')
    a, b = features(3), features(4)
    ours = knn_null(a, b, 10, permutations=40, seed=9)
    theirs = original.measure(torch.from_numpy(a).float(), torch.from_numpy(b).float(), 10, 9, 40)
    assert ours['score'] == pytest.approx(theirs['score'], abs=0.01)
    assert ours['null_mean'] == pytest.approx(theirs['null_mean'], abs=0.01)
