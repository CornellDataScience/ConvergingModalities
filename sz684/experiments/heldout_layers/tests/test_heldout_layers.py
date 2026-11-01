from shared.paths import PROJECT_ROOT, project_path
import numpy as np
import torch
from experiments.baseline.evaluation.alignment import neighbors, overlap
from experiments.heldout_layers.evaluation.heldout_layers import (
    fit_threshold,
    layer_matrix,
)


def test_threshold_ignores_evaluation_rows():
    raw = torch.randn(12, 2, 7, generator=torch.Generator().manual_seed(34))
    expected = fit_threshold(raw, list(range(6)), 0.95)
    raw[6:] *= 10000
    assert torch.equal(expected, fit_threshold(raw, list(range(6)), 0.95))


def test_layer_matrix_matches_set_intersections():
    rng = np.random.default_rng(12)
    left = [neighbors(torch.tensor(rng.normal(size=(20, 8))), 3) for _ in range(2)]
    right = [neighbors(torch.tensor(rng.normal(size=(20, 9))), 3) for _ in range(3)]
    actual = layer_matrix(left, right)
    expected = [
        [np.mean([len(set(x) & set(y)) / 3 for x, y in zip(a, b)]) for b in right]
        for a in left
    ]
    np.testing.assert_allclose(actual, expected)
