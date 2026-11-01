"""Known transformations check scientific meanings, not implementation details."""
import numpy as np
import pytest
from ..evaluation.rsa import rsa
from ..evaluation.cka import linear_cka
from ..evaluation.neighbors import neighbors, neighbor_scores
from ..training.alignment import fit_procrustes, map_endpoints
from ..training.probes import fit_probe
from ..evaluation.directions import evaluate_transfer
from ..evaluation.retrieval import retrieve


def test_geometry_under_rotation_and_scaling():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    y = 3*x@rotation
    assert rsa(x, y) == pytest.approx(1)
    assert linear_cka(x, y+10) == pytest.approx(1)
    assert neighbor_scores(x, y, 3)['jaccard'].mean() == 1
    assert linear_cka(x, np.c_[y, np.zeros((30, 2))]) == pytest.approx(1)


def test_invalid_and_degenerate_features():
    with pytest.raises(ValueError):
        rsa(np.ones((5, 3)), np.ones((5, 3)))
    with pytest.raises(ValueError):
        linear_cka(np.ones((5, 3)), np.ones((5, 3)))
    with pytest.raises(ValueError):
        neighbors(np.eye(3), 3)


def test_neighbors_exclude_self_and_ties_are_stable():
    actual = neighbors(np.eye(4), 2)
    assert actual.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1]]


def test_transport_classifier_origin_and_signed_normal():
    rng = np.random.default_rng(7)
    training = rng.normal(size=(80, 4))
    test = rng.normal(size=(30, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    target = training@rotation + 8
    fitted = fit_procrustes(training, target)
    np.testing.assert_allclose(map_endpoints(test, fitted), test@rotation+8, atol=1e-12)
    source_probe = fit_probe(training, (training[:, 0] > 0).astype(int))
    target_probe = fit_probe(target, (training[:, 0] > 0).astype(int))
    score = evaluate_transfer(source_probe, target_probe, fitted, test@rotation+8, (test[:,0] > 0).astype(int))
    assert score['cav_cosine'] == pytest.approx(1)
    assert score['transfer_balanced_accuracy'] == score['target_balanced_accuracy']


def test_retrieval_excludes_source_and_preserves_identity():
    features = np.array([[1.,0,0], [1,1,0], [0,0,1], [0,1,1]])
    rows = [{'source_id':str(i), 'family_id':str(i//2), 'label':str(i%2)} for i in range(4)]
    shifted = retrieve(features, rows, np.array([0.,1,0]), 1)
    assert shifted['ap'] == shifted['joint'] == shifted['identity'] == 1
