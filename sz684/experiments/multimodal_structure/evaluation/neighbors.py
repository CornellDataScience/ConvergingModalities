"""Labeled neighborhood agreement, rather than unlabeled graph isomorphism."""
import numpy as np
from .rsa import normalize_rows


def neighbors(features, k):
    """Return [concept, neighbor] indices, excluding self; ties use row order."""
    unit = normalize_rows(features)
    if not 0 < k < len(unit):
        raise ValueError('Require 0 < k < number of concepts')
    similarities = unit @ unit.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argsort(-similarities, axis=1, kind='stable')[:, :k]


def neighbor_scores(left, right, k):
    """Return per-concept Jaccard and intersection/k for corresponding rows."""
    if len(left) != len(right):
        raise ValueError('Concept counts differ')
    a, b = neighbors(left, k), neighbors(right, k)
    intersection = np.array([len(set(x) & set(y)) for x, y in zip(a, b)])
    return {'jaccard': intersection / (2 * k - intersection),
            'overlap': intersection / k}
