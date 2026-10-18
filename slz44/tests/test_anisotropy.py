import numpy as np
import pytest
import torch
import torch.nn.functional as F
from anisotropy import apply_control, diagnostics, fit_control, measure, neighbors, overlap


@pytest.fixture
def cone():
    """Anisotropic features: a shared offset plus one rogue dimension, L2-normalized."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(300, 32, generator=generator)
    x[:, 0] *= 20
    return F.normalize(x + 5., dim=-1)


def test_baseline_is_identity_and_center_matches_definition(cone):
    assert torch.equal(apply_control(cone, fit_control(cone, 'baseline')), cone)
    expected = F.normalize(cone - cone.mean(0), dim=-1)
    assert torch.allclose(apply_control(cone, fit_control(cone, 'center')), expected)


def test_pca128_matches_definition_and_reports_retained_variance(cone):
    fitted = fit_control(cone, 'pca128', dimensions=8)
    _, _, vh = torch.linalg.svd(cone - cone.mean(0), full_matrices=False)
    expected = F.normalize((cone - cone.mean(0)) @ vh[:8].T, dim=-1)
    assert torch.allclose(apply_control(cone, fitted), expected, atol=1e-5)
    assert 0 < fitted['retained'] <= 1


def test_outputs_unit_norm(cone):
    for mode in ['center', 'pca128', 'standardize', 'remove3', 'pca128_whiten']:
        out = apply_control(cone, fit_control(cone, mode))
        assert torch.allclose(out.norm(dim=1), torch.ones(300), atol=1e-5)


def test_standardize_scale_matches_selection_std(cone):
    fitted = fit_control(cone, 'standardize')
    assert torch.allclose(fitted['scale'], (cone - cone.mean(0)).std(0), atol=1e-6)


def test_remove3_projects_out_top_directions(cone):
    fitted = fit_control(cone, 'remove3')
    out = apply_control(cone, fitted)
    assert out.shape[1] == cone.shape[1]
    assert float((out @ fitted['basis']).abs().max()) < 1e-5


def test_whitening_gives_identity_covariance(cone):
    fitted = fit_control(cone, 'pca128_whiten', dimensions=8)
    z = ((cone - fitted['mean']) @ fitted['basis']) / fitted['scale']
    assert torch.allclose(torch.cov(z.T), torch.eye(8), atol=1e-4)


def test_unknown_mode_raises(cone):
    with pytest.raises(ValueError):
        fit_control(cone, 'nope')


def test_neighbors_exclude_self_and_overlap_identity():
    x = F.normalize(torch.randn(60, 8, generator=torch.Generator().manual_seed(2)), dim=-1)
    found = neighbors(x, 5)
    assert not (found == np.arange(60)[:, None]).any()
    assert overlap(found, found) == 1.0


def test_measure_identical_features_beats_null():
    x = F.normalize(torch.randn(60, 8, generator=torch.Generator().manual_seed(3)), dim=-1)
    result = measure(x, x, 5, seed=0, permutations=200)
    assert result['score'] == 1.0
    assert abs(result['null_mean'] - 5 / 59) < .03
    assert result['p'] == pytest.approx(1 / 201)


def test_diagnostics_detect_cone_and_controls_reduce_it(cone):
    raw = diagnostics(cone)
    assert raw['mean_cosine'] > .3
    for mode in ['center', 'standardize', 'remove3']:
        assert diagnostics(apply_control(cone, fit_control(cone, mode)))['mean_cosine'] < raw['mean_cosine']
    isotropic = F.normalize(torch.randn(300, 32, generator=torch.Generator().manual_seed(1)), dim=-1)
    assert abs(diagnostics(isotropic)['mean_cosine']) < .05
    assert diagnostics(isotropic)['effective_rank'] > raw['effective_rank']