import numpy as np
import pytest
from extra_metrics import (cka_with_null, gw_cost, gw_cost_with_null, gw_match, linear_cka,
                           normalized_distance)

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


def test_cka_identity_orthogonal_invariance_and_offset():
    a, b = pair()
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 16)))
    assert linear_cka(a, a) == pytest.approx(1.0)
    assert linear_cka(a, a @ q) == pytest.approx(1.0)
    assert linear_cka(a, a + 50.0) == pytest.approx(1.0)      # constant offset ignored
    assert 0 < linear_cka(a, b) < 1


def test_cka_null_beats_shuffled_pairings():
    a, b = pair()
    result = cka_with_null(a, b, permutations=50)
    assert result['excess'] > 0.05 and len(result['null']) == 50


def test_gw_cost_zero_for_identical_and_reduction_positive():
    a, b = pair()
    c1, c2 = normalized_distance(a), normalized_distance(b)
    assert gw_cost(c1, c1) == 0
    assert c1.shape == (120, 120) and np.allclose(np.diag(c1), 0)
    result = gw_cost_with_null(c1, c2, permutations=50)
    assert result['reduction'] > 0.1


def test_gw_cost_reduction_near_zero_for_independent():
    rng = np.random.default_rng(3)
    c1 = normalized_distance(rng.standard_normal((120, 16)))
    c2 = normalized_distance(rng.standard_normal((120, 16)))
    assert abs(gw_cost_with_null(c1, c2, permutations=50)['reduction']) < 0.05


@pytest.mark.skipif(not HAVE_OT, reason='POT not installed')
def test_gw_match_recovers_a_rotated_copy():
    a, _ = pair(n=80)
    q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((16, 16)))
    c1, c2 = normalized_distance(a), normalized_distance(a @ q)
    result = gw_match(c1, c2)
    assert result['top1'] > 0.5
