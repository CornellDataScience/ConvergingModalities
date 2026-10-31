import numpy as np
import pytest
from prh.controls import MODES, apply_control, clip_normalize, diagnostics, fit_control, l2_normalize


@pytest.fixture
def cone():
    """Anisotropic features: a shared offset plus one rogue dimension, L2-normalized."""
    x = np.random.default_rng(0).standard_normal((300, 32))
    x[:, 0] *= 20
    return l2_normalize(x + 5.)


def test_baseline_is_identity_and_center_matches_definition(cone):
    assert np.array_equal(apply_control(cone, fit_control(cone, 'baseline')), cone)
    expected = l2_normalize(cone - cone.mean(0))
    assert np.allclose(apply_control(cone, fit_control(cone, 'center')), expected)


def test_pca128_matches_definition_and_reports_retained_variance(cone):
    fitted = fit_control(cone, 'pca128', dimensions=8)
    _, _, vh = np.linalg.svd(cone - cone.mean(0), full_matrices=False)
    expected = l2_normalize((cone - cone.mean(0)) @ vh[:8].T)
    assert np.allclose(apply_control(cone, fitted), expected, atol=1e-9)
    assert 0 < fitted['retained'] <= 1


def test_outputs_unit_norm(cone):
    for mode in MODES[1:]:
        out = apply_control(cone, fit_control(cone, mode))
        assert np.allclose(np.linalg.norm(out, axis=1), 1)


def test_standardize_scale_matches_selection_std(cone):
    fitted = fit_control(cone, 'standardize')
    assert np.allclose(fitted['scale'], (cone - cone.mean(0)).std(0, ddof=1))


def test_remove3_projects_out_top_directions(cone):
    fitted = fit_control(cone, 'remove3')
    out = apply_control(cone, fitted)
    assert out.shape[1] == cone.shape[1]
    assert np.abs(out @ fitted['basis']).max() < 1e-9


def test_whitening_gives_identity_covariance(cone):
    fitted = fit_control(cone, 'pca128_whiten', dimensions=8)
    z = ((cone - fitted['mean']) @ fitted['basis']) / fitted['scale']
    assert np.allclose(np.cov(z.T), np.eye(8), atol=1e-8)


def test_controls_are_fitted_on_training_rows_only(cone):
    fitted = fit_control(cone[:200], 'center')
    assert np.allclose(fitted['mean'], cone[:200].mean(0))
    assert apply_control(cone[200:], fitted).shape == cone[200:].shape


def test_unknown_mode_raises(cone):
    with pytest.raises(ValueError):
        fit_control(cone, 'nope')


def test_clip_normalize_limits_rogue_values():
    x = np.array([[100., 1., 1.], [1., 1., 1.]])
    out = clip_normalize(x, 2.0)
    assert np.allclose(np.linalg.norm(out, axis=1), 1)
    assert out[0, 0] == pytest.approx(2 / np.sqrt(6))


def test_diagnostics_detect_cone_and_controls_reduce_it(cone):
    raw = diagnostics(cone)
    assert raw['mean_cosine'] > .3
    for mode in ['center', 'standardize', 'remove3']:
        assert diagnostics(apply_control(cone, fit_control(cone, mode)))['mean_cosine'] < raw['mean_cosine']
    isotropic = l2_normalize(np.random.default_rng(1).standard_normal((300, 32)))
    assert abs(diagnostics(isotropic)['mean_cosine']) < .05
    assert diagnostics(isotropic)['effective_rank'] > raw['effective_rank']
