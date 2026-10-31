import numpy as np
import pytest
from prh.controls import l2_normalize
from prh.metrics import gw_cost, gw_match, linear_cka, mutual_knn, neighbors, normalized_distance, overlap
from prh.nulls import cka_null, gw_cost_null, knn_null

try:
    import ot  # noqa: F401
    HAVE_OT = True
except ImportError:
    HAVE_OT = False


def pair(n=120, d=16, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 6))
    a = latent @ rng.standard_normal((6, d)) + noise * rng.standard_normal((n, d))
    b = latent @ rng.standard_normal((6, d + 4)) + noise * rng.standard_normal((n, d + 4))
    return a, b


def test_neighbors_exclude_self_and_overlap_identity():
    x = l2_normalize(np.random.default_rng(2).standard_normal((60, 8)))
    found = neighbors(x, 5)
    assert not (found == np.arange(60)[:, None]).any()
    assert overlap(found, found) == 1.0
    with pytest.raises(ValueError):
        neighbors(x, 60)


def test_knn_null_identical_features_beats_null():
    x = l2_normalize(np.random.default_rng(3).standard_normal((60, 8)))
    result = knn_null(x, x, 5, permutations=200, seed=0)
    assert result['score'] == 1.0 == mutual_knn(x, x, 5)
    assert abs(result['null_mean'] - 5 / 59) < .03
    assert result['p'] == pytest.approx(1 / 201)


def test_knn_null_related_spaces_have_positive_excess():
    a, b = pair()
    assert knn_null(l2_normalize(a), l2_normalize(b), 10, permutations=50)['excess'] > 0.05


def test_cka_identity_orthogonal_invariance_and_offset():
    a, b = pair()
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 16)))
    assert linear_cka(a, a) == pytest.approx(1.0)
    assert linear_cka(a, a @ q) == pytest.approx(1.0)
    assert linear_cka(a, a + 50.0) == pytest.approx(1.0)
    assert 0 < linear_cka(a, b) < 1


def test_cka_null_beats_shuffled_pairings():
    a, b = pair()
    result = cka_null(a, b, permutations=50)
    assert result['excess'] > 0.05 and len(result['null']) == 50


def test_gw_cost_zero_for_identical_and_reduction_positive():
    a, b = pair()
    c1 = normalized_distance(a)
    assert gw_cost(c1, c1) == 0
    assert c1.shape == (120, 120) and np.allclose(np.diag(c1), 0)
    assert gw_cost_null(a, b, permutations=50)['reduction'] > 0.1


def test_gw_cost_reduction_near_zero_for_independent():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((120, 16)), rng.standard_normal((120, 16))
    assert abs(gw_cost_null(a, b, permutations=50)['reduction']) < 0.05


def test_nulls_are_reproducible_for_a_seed():
    a, b = pair()
    assert cka_null(a, b, 20, seed=5)['null'] == cka_null(a, b, 20, seed=5)['null']


@pytest.mark.skipif(not HAVE_OT, reason='POT not installed')
def test_gw_match_recovers_a_rotated_copy():
    a, _ = pair(n=80)
    q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((16, 16)))
    result = gw_match(normalized_distance(a), normalized_distance(a @ q))
    assert result['top1'] > 0.5
