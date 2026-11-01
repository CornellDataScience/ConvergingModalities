from shared.paths import PROJECT_ROOT, project_path
import numpy as np
import torch
from experiments.color_transfer.data.quickdraw_data import render_mask, recolor
from shared.data import bank_allowed, fit_preprocessing
from experiments.color_transfer.evaluation.semantic_analysis import (
    bootstrap_paired,
    retrieval,
    choose_scale,
)


def test_counterfactual_changes_only_mask_and_preserves_channel_sum():
    mask = render_mask([[[0, 255, 0], [0, 255, 255]]])
    red = np.array(recolor(mask, [200, 40, 40], 245))
    blue = np.array(recolor(mask, [40, 40, 200], 245))
    assert np.array_equal(red[..., 1], blue[..., 1])
    assert np.array_equal(red[..., 0], blue[..., 2])
    assert not np.any(red[np.array(mask) == 0] != blue[np.array(mask) == 0])


def test_pair_shuffle_leaves_mean_relation_unchanged():
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(8, 4, dtype=torch.float64)
    torch.testing.assert_close((y.flip(0) - x).mean(0), (y - x).mean(0))


def test_retrieval_bank_is_balanced_and_keeps_counterfactual():
    c, n = 3, 4
    mask = bank_allowed(c, n)
    assert np.all(mask.sum(1) == c * (n - 1) * 2)
    assert np.all(mask[:, 0::2].sum(1) == mask[:, 1::2].sum(1))
    assert np.all(mask[np.arange(c * n), np.arange(c * n) * 2] == False)
    assert np.all(mask[np.arange(c * n), np.arange(c * n) * 2 + 1])


def test_retrieval_ranks_and_identity_exclusion_against_brute_force():
    generator = torch.Generator().manual_seed(75)
    y = torch.nn.functional.normalize(
        torch.randn(2, 4, 2, 8, generator=generator, dtype=torch.float64), dim=-1
    )
    q = y[:, :, 1]
    actual = retrieval(q, y)
    assert np.all(actual["global"]["paired_rank"] == 1)
    assert np.all(actual["global"]["paired_top1"])
    assert np.all(actual["within_category"]["paired_top1"])


def test_cluster_bootstrap_preserves_pairing_and_known_constant():
    effects = np.stack([np.ones((3, 4)) * 2, np.ones((3, 4)) * -3])
    boot, category_only, _, _ = bootstrap_paired(effects, 1, 100)
    np.testing.assert_allclose(boot[0], 2)
    np.testing.assert_allclose(boot[1], -3)
    np.testing.assert_allclose(category_only, boot)


def test_preprocessing_ignores_heldout_outliers():
    raw = torch.randn(
        20, 8, generator=torch.Generator().manual_seed(76), dtype=torch.float64
    )
    quantiles = torch.quantile(raw.abs(), 0.95, dim=1)
    f, meta = fit_preprocessing(raw, quantiles, list(range(12)), 4)
    expected = f(raw[:12])
    raw[12:] *= 10000
    quantiles[12:] *= 10000
    g, new = fit_preprocessing(raw, quantiles, list(range(12)), 4)
    torch.testing.assert_close(expected, g(raw[:12]))
    assert meta == new


def test_scale_selection_uses_validation_and_zero_tie_break():
    x = torch.eye(3, dtype=torch.float64)
    scale, values = choose_scale(x, x, torch.zeros(3, dtype=torch.float64), [0, 0.5, 1])
    assert scale == 0 and values == [0, 0, 0]
