"""Small serializable train-only transforms for alignment and probes."""
import numpy as np


def fit_pca(training, dimensions):
    """Fit centering and an orthonormal basis using training anchors only."""
    training = np.asarray(training, dtype=np.float64)
    if training.ndim != 2 or not np.isfinite(training).all():
        raise ValueError('Expected finite training matrix')
    mean = training.mean(axis=0)
    _, singular, basis = np.linalg.svd(training - mean, full_matrices=False)
    rank = np.sum(singular > max(singular[0], 1e-12) * 1e-8)
    if dimensions < 1 or dimensions > rank:
        raise ValueError(f'Requested PCA width {dimensions}; effective rank is {rank}')
    return {'mean': mean, 'basis': basis[:dimensions].T}


def transform(features, fitted):
    """Apply the saved training mean and basis; no whitening or renormalizing."""
    return (np.asarray(features) - fitted['mean']) @ fitted['basis']
