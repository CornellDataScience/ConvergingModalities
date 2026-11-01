"""Orthogonal maps use row vectors throughout: target approximately source @ R."""
import numpy as np


def fit_procrustes(source, target):
    """Fit a centered square orthogonal map on corresponding training anchors."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError('Procrustes requires equal [anchors, dimensions] shapes')
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-source_mean).T @ (target-target_mean))
    return {'rotation': u @ vt, 'source_mean': source_mean, 'target_mean': target_mean}


def map_endpoints(features, fitted):
    """Endpoint prediction includes centering and translation."""
    return (features - fitted['source_mean']) @ fitted['rotation'] + fitted['target_mean']


def map_direction(direction, fitted):
    """For this orthogonal map only, normals and displacements share this rule."""
    return direction @ fitted['rotation']
