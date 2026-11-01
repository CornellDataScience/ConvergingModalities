from shared.paths import PROJECT_ROOT, project_path
import torch


def test_identity_projection_removes_only_training_subspace_component():
    from experiments.state_transfer.evaluation.state_identity_followup import (
        project_direction,
    )

    d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    basis = torch.eye(3, dtype=torch.float64)[:, :1]
    p = project_direction(d, basis, 1)
    assert p[0] == 0
    torch.testing.assert_close(p.norm(), d.norm())
    torch.testing.assert_close(project_direction(d, basis, 0), d)
