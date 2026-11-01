from shared.paths import PROJECT_ROOT, project_path
import numpy as np
import torch
from scipy.stats import bootstrap
from experiments.paired_uncertainty.evaluation.paired_uncertainty import (
    reference_overlap,
    intervals,
)


def test_reference_overlap_against_independent_sets():
    rng = np.random.default_rng(91)
    a = rng.normal(size=(270, 8))
    b = rng.normal(size=(270, 9))
    na = np.argsort(-(a[256:] @ a[:128].T), axis=1)[:, :5]
    nb = np.argsort(-(b[256:] @ b[:128].T), axis=1)[:, :5]
    expected = [len(set(x) & set(y)) / 5 for x, y in zip(na, nb)]
    np.testing.assert_allclose(
        reference_overlap(torch.tensor(a), torch.tensor(b), 5, 128), expected
    )
    np.testing.assert_allclose(
        reference_overlap(torch.tensor(a), torch.tensor(a), 5, 128), np.ones(14)
    )


def test_difference_bootstrap_equals_explicit_paired_bootstrap():
    rng = np.random.default_rng(32)
    a = rng.normal(size=80)
    b = 0.8 * a + rng.normal(size=80) * 0.2
    actual = intervals(a - b, 19, resamples=1000)
    expected = bootstrap(
        (a, b),
        lambda a, b, axis: np.mean(a - b, axis=axis),
        paired=True,
        vectorized=True,
        n_resamples=1000,
        method="BCa",
        rng=np.random.default_rng(19),
        batch=500,
    )
    np.testing.assert_allclose(actual["draws"], expected.bootstrap_distribution)
    np.testing.assert_allclose(actual["ci95"], expected.confidence_interval)
