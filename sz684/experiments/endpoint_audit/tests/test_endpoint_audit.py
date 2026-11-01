from shared.paths import PROJECT_ROOT, project_path
import torch
from experiments.endpoint_audit.evaluation.endpoint_audit import (
    direction_details,
    project_directions,
)


def test_full_dimensional_rank_is_not_projected_rank():
    target = torch.eye(4, dtype=torch.float64)
    predicted = target[[1, 0, 2, 3]]
    similarities, ranks = direction_details(predicted, target)
    assert ranks.tolist() == [2, 2, 1, 1]
    torch.testing.assert_close(
        similarities.diag(), torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    )


def test_joint_projection_energy_and_pair_order():
    # A rank-two dataset should be preserved exactly up to a rotation.
    predicted = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    projected, retention = project_directions(predicted, target)
    combined = torch.cat([predicted, target])
    torch.testing.assert_close(projected @ projected.T, combined @ combined.T)
    assert abs(retention - 1.0) < 1e-12
