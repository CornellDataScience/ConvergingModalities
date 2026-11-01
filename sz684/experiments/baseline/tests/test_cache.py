from shared.paths import PROJECT_ROOT, project_path
import json
import numpy as np
import pytest
from experiments.baseline.data.experiment import load_features


def test_reordered_or_missing_rows_rejected(tmp_path):
    spec = {"id": "test/model"}
    folder = tmp_path / "features" / "test__model"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"complete": True, "identity": {"sample_hash": "abc"}})
    )
    np.savez(folder / "0000.npz", feats=np.ones((2, 1, 3)), rows=[1, 0])
    with pytest.raises(ValueError, match="reordered"):
        load_features(tmp_path, spec, {"sample_hash": "abc", "samples": 2})
