"""Evaluate a trained CAV without treating it as a centroid displacement."""
import numpy as np
from ..training.alignment import map_direction
from ..training.probes import predict, balanced_accuracy


def signed_cosine(left, right):
    """Preserve the source-to-target sign; reversed axes should score -1."""
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        raise ValueError('Cannot compare a zero direction')
    return float(np.clip(left @ right / denominator, -1, 1))


def evaluate_transfer(source_probe, target_probe, alignment, target_test, labels):
    """Compare normals and evaluate the source classifier in target coordinates.

    A transported classifier retains its transported decision-plane origin;
    copying the target probe's mean would leak target supervision into it.
    """
    normal = map_direction(source_probe['normal'], alignment)
    mean = ((source_probe['mean'] - alignment['source_mean']) @ alignment['rotation']
            + alignment['target_mean'])
    transported = {'normal': normal, 'mean': mean}
    return {'cav_cosine': signed_cosine(normal, target_probe['normal']),
            'transfer_balanced_accuracy': balanced_accuracy(labels, predict(target_test, transported)),
            'target_balanced_accuracy': balanced_accuracy(labels, predict(target_test, target_probe))}
