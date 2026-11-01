from shared.paths import PROJECT_ROOT, project_path
import numpy as np
import pytest
import torch
from experiments.baseline.evaluation.alignment import (
    neighbors,
    overlap,
    prepare,
    score_layers,
)


def test_identity_and_reference():
    x = prepare(torch.randn(50, 2, 13, generator=torch.Generator().manual_seed(1)), q=1)
    a, b = neighbors(x[:, 0], 5), neighbors(x[:, 1], 5)
    expected = np.mean([len(set(u) & set(v)) / 5 for u, v in zip(a, b)])
    assert overlap(a, b) == pytest.approx(expected)
    result = score_layers(x[:, :1], x[:, 1:], 5, 3, seed=91)
    assert result["best_score"] == pytest.approx(expected)
    rng = np.random.default_rng(91)
    explicit_null = []
    for _ in range(3):
        p = rng.permutation(len(x))
        explicit_null.append(overlap(a, neighbors(x[p, 1], 5)))
    assert result["permuted_maxima"] == pytest.approx(explicit_null)
    assert overlap(a, a) == 1
    assert not (a == np.arange(50)[:, None]).any()
    assert score_layers(x, x, 5, 0)["best_score"] == 1


def test_joint_permutation():
    x = prepare(torch.randn(60, 2, 17, generator=torch.Generator().manual_seed(2)), q=1)
    p = np.random.default_rng(3).permutation(len(x))
    original = score_layers(x[:, :1], x[:, 1:], 5, 0)["best_score"]
    permuted = score_layers(x[p, :1], x[p, 1:], 5, 0)["best_score"]
    assert original == permuted


def test_permutation_remapping_matches_recomputation():
    x = prepare(torch.randn(50, 1, 11, generator=torch.Generator().manual_seed(4)), q=1)
    p = np.random.default_rng(5).permutation(len(x))
    n = neighbors(x[:, 0], 5)
    assert np.array_equal(np.argsort(p)[n[p]], neighbors(x[p, 0], 5))


def test_clipping_matches_upstream():
    x = torch.randn(20, 3, 9, generator=torch.Generator().manual_seed(6))
    for mode in ["upstream_default", "upstream_precise"]:
        threshold = (
            torch.quantile(x.abs().flatten(1), 0.95, dim=1).mean()
            if mode == "upstream_default"
            else x.abs().flatten().sort().values[int(0.95 * x.numel())]
        )
        expected = torch.nn.functional.normalize(x.clamp(-threshold, threshold), dim=-1)
        assert torch.equal(prepare(x, mode=mode), expected)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        neighbors(torch.eye(3), 3)
    with pytest.raises(ValueError):
        prepare(torch.full((3, 1, 2), float("nan")))


def test_masked_pooling():
    from shared.io import pooled

    hidden = torch.tensor([[[99.0, 99.0], [2.0, 4.0], [4.0, 8.0]]])
    mask = torch.tensor([[0, 1, 1]])
    assert torch.equal(pooled(hidden, mask), torch.tensor([[3.0, 6.0]]))
