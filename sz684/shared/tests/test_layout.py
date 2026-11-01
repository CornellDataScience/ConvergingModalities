"""Folder migration must preserve access to historical artifacts and cache safety."""

import json
from pathlib import Path

import pytest
import torch

from shared.cache import migrated_legacy_hash, resume_identity
from shared.paths import PROJECT_ROOT, project_path
from shared.training import fit_ridge
from experiments.state_transfer.training.bridge import fit_bridge


def test_paths_resolve_old_and_canonical_names_outside_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert project_path("results/state_transfer/results.json") == project_path(
        "experiments/state_transfer/results/results.json"
    )
    assert project_path("state_analysis.py") == project_path(
        "experiments/state_transfer/evaluation/state_analysis.py"
    )
    assert project_path(tmp_path / "custom") == tmp_path / "custom"
    assert (
        project_path("results/baseline")
        == PROJECT_ROOT / "experiments/baseline/results/baseline"
    )


def test_cache_migration_preserves_identity_and_rejects_setting_changes():
    current = dict(model="frozen-model", batch_size=16, extractor_hash="new")
    cached = dict(current, extractor_hash="known-old")
    assert resume_identity(cached, current, "known-old") == cached
    assert resume_identity(current, current, None) == current
    for invalid in [dict(cached, batch_size=8), dict(cached, extractor_hash="unknown")]:
        with pytest.raises(ValueError, match="identity mismatch"):
            resume_identity(invalid, current, "known-old")
    with pytest.raises(ValueError, match="identity mismatch"):
        resume_identity(cached, current, None)


@pytest.mark.parametrize(
    "experiment,filename",
    [("state_transfer", "state_extract.py"), ("color_transfer", "semantic_extract.py")],
)
def test_cache_migration_is_pinned_to_exact_reorganized_source(
    experiment, filename, tmp_path
):
    source = PROJECT_ROOT / "experiments" / experiment / "data" / filename
    assert migrated_legacy_hash(source, "known-old") == "known-old"
    changed = tmp_path / filename
    changed.write_bytes(source.read_bytes() + b"\n# changed extractor\n")
    assert migrated_legacy_hash(changed, "known-old") is None


def test_separated_training_bridge_matches_direct_solve():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    y = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    w, xm, ym, orthogonal = fit_bridge(x, y, 1.0)
    torch.testing.assert_close(w, fit_ridge(x, y, 1.0), rtol=0, atol=0)
    torch.testing.assert_close(xm, x.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(ym, y.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(orthogonal.T @ orthogonal, torch.eye(4, dtype=x.dtype))
