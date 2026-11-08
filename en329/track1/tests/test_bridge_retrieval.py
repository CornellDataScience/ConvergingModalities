import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge_retrieval as br


def test_ridge_recovers_affine_map_and_solver_matches_refit():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 7))
    m = rng.normal(size=(7, 5))
    y = x @ m + np.arange(5)
    np.testing.assert_allclose(br.fit_ridge(x, y, alpha=0.0), m, atol=1e-10)
    p = rng.permutation(60)
    np.testing.assert_allclose(br.ridge_solver(x) @ (y[p] - y.mean(0)), br.fit_ridge(x, y[p]), atol=1e-10)


def test_score_ranks_excludes_self_and_reports_retention():
    # Gallery: 0 = (a, raw) query, 1 = (a, cooked) target, 2 = (b, cooked), 3 = (a, raw) another source image
    g = np.eye(4)
    obj = np.array(['a', 'a', 'b', 'a'])
    state = np.array(['raw', 'cooked', 'cooked', 'raw'])
    q = np.array([[1.0, 0.1, 0.5, 0.0]])  # self is most similar, then wrong object, then target
    r = br.score(q, g, obj, state, np.array(['a']), np.array(['cooked']), np.array([0]))
    assert r['joint_rr'][0] == pytest.approx(1 / 2)  # self excluded; (b, cooked) ranks above (a, cooked)
    assert r['joint_at1'][0] == 0 and r['joint_at5'][0] == 1
    assert r['object_at1'][0] == 0 and r['state_at1'][0] == 1


def test_score_without_any_target_gives_zero():
    g = np.eye(3)
    r = br.score(g[:1], g, np.array(['a', 'a', 'b']), np.array(['raw', 'raw', 'cooked']),
                 np.array(['a']), np.array(['cooked']), np.array([0]))
    assert r['joint_rr'][0] == 0 and r['rank'][0] == -1


def test_cluster_bootstrap_constant_difference_and_clustering():
    d = np.full(30, 0.25)
    r = br.cluster_bootstrap(d, np.repeat(np.arange(6), 5), draws=500, family=4)
    assert r['effect'] == pytest.approx(0.25) and r['ci_adjusted'] == pytest.approx([0.25, 0.25])
    assert r['clusters'] == 6 and r['queries'] == 30


def test_exact_spearman_perfect_order():
    rho, p = br.exact_spearman(np.arange(5.0), np.arange(5.0) * 2)
    assert rho == pytest.approx(1.0) and p == pytest.approx(2 / 120)


def test_alignment_margin():
    v = {'a': np.array([1.0, 0, 0]), 'b': np.array([0, 1.0, 0])}
    r = br.alignment(v, v, ['a', 'b'])
    assert r['margin'] == pytest.approx([1.0, 1.0])


def test_overlapping_splits_rejected():
    data, split = br.synthetic(0)
    split['test'] = split['test'] + split['train'][:1]
    with pytest.raises(AssertionError):
        br.run(data, split, shuffles=2, calibration_draws=1, bootstrap_draws=10)


def test_directions_use_only_training_objects():
    data, split = br.synthetic(0)
    test = set(split['test'])
    # Corrupt every test-object text and image; directions/alignment fitted on train must not change.
    poisoned = {k: v.copy() for k, v in data.items()}
    for side in ('image', 'text'):
        m = np.isin(poisoned[f'{side}_object'], list(test))
        poisoned[f'{side}_X'][m] = np.random.default_rng(0).normal(size=(m.sum(), poisoned[f'{side}_X'].shape[1]))
    a = br.run(data, split, shuffles=2, calibration_draws=1, bootstrap_draws=10)['alignment']['matrix']
    b = br.run(poisoned, split, shuffles=2, calibration_draws=1, bootstrap_draws=10)['alignment']['matrix']
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize('null_text,expected', [(False, True), (True, False)])
def test_end_to_end_signal_passes_and_null_fails(null_text, expected):
    data, split = br.synthetic(0, null_text)
    r = br.run(data, split, shuffles=40, calibration_draws=3, bootstrap_draws=2000)
    assert r['verdict']['transfer_supported'] is expected
    assert r['summary']['no_change']['overall']['object_at1'] == 1.0
