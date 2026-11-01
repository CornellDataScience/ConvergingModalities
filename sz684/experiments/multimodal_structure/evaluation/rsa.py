"""Compare concept distance orderings without aligning feature coordinates."""
import numpy as np
from scipy.stats import spearmanr


def normalize_rows(features):
    """Return unit-length rows of a finite [examples, coordinates] matrix."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Features must be a finite two-dimensional matrix')
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError('Zero-length features have no cosine direction')
    return features / lengths


def distance_matrix(features):
    """Calculate cosine distances; each row must represent the same concept."""
    unit = normalize_rows(features)
    distances = np.clip(1 - unit @ unit.T, 0, 2)
    np.fill_diagonal(distances, 0)
    return distances


def rsa(left, right):
    """Spearman correlation of the strict upper triangles; no naive p-value."""
    if len(left) != len(right) or len(left) < 3:
        raise ValueError('RSA needs at least three corresponding concepts')
    upper = np.triu_indices(len(left), k=1)
    a = distance_matrix(left)[upper]
    b = distance_matrix(right)[upper]
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        raise ValueError('RSA is undefined for constant distances')
    return float(spearmanr(a, b).statistic)
