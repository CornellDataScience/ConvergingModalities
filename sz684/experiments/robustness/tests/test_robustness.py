from shared.paths import PROJECT_ROOT, project_path
import numpy as np
import torch
from experiments.baseline.evaluation.alignment import neighbors, overlap
from shared.transforms import fit_transform, apply_transform
from experiments.robustness.evaluation.robustness import measure


def test_center_uses_training_mean():
    train = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    evaluation = torch.tensor([[2.0, 4.0, 2.0]])
    fit = fit_transform(train, "center")
    expected = torch.nn.functional.normalize(
        evaluation - torch.tensor([2.0, 3.0, 0.0]), dim=-1
    )
    torch.testing.assert_close(apply_transform(evaluation, fit), expected)
    assert apply_transform(evaluation, fit_transform(train, "baseline")) is evaluation


def test_pca_matches_training_covariance_subspace():
    train = torch.randn(40, 7, generator=torch.Generator().manual_seed(43))
    fit = fit_transform(train, "pca128", dimensions=3)
    centered = train - train.mean(0)
    eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered)
    expected = eigenvectors[:, -3:]
    torch.testing.assert_close(
        fit[1] @ fit[1].T, expected @ expected.T, atol=2e-5, rtol=2e-5
    )
    assert np.isclose(
        fit[2], float(eigenvalues[-3:].sum() / eigenvalues.sum()), atol=1e-6
    )
    heldout = torch.randn(9, 7, generator=torch.Generator().manual_seed(44))
    assert apply_transform(heldout, fit).shape == (9, 3)
    torch.testing.assert_close(apply_transform(heldout, fit).norm(dim=1), torch.ones(9))


def test_measure_null_matches_explicit_recomputation():
    generator = torch.Generator().manual_seed(45)
    a = torch.nn.functional.normalize(torch.randn(24, 7, generator=generator), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(24, 9, generator=generator), dim=-1)
    actual = measure(a, b, 3, 92, permutations=5)
    rng = np.random.default_rng(92)
    left = neighbors(a, 3)
    expected = [overlap(left, neighbors(b[rng.permutation(24)], 3)) for _ in range(5)]
    np.testing.assert_allclose(actual["null"], expected)
    assert actual["score"] == overlap(left, neighbors(b, 3))
