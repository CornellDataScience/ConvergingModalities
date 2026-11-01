"""A ridge binary classifier gives a simple, regularized CAV baseline.

This is least-squares classification, not logistic regression or full TCAV.
Positive labels always identify the target endpoint of the relation.
"""
import numpy as np


def fit_probe(features, labels, alpha=1.0):
    """Fit a class-balanced ridge classifier on [examples, coordinates] rows."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels) or set(labels) != {0, 1}:
        raise ValueError('Need matching rows and both binary labels 0 and 1')
    if alpha <= 0 or not np.isfinite(features).all():
        raise ValueError('Need finite features and positive regularization')
    # Give both classes equal total weight, even if their counts differ.
    weights = np.array([len(labels)/(2*np.sum(labels == label)) for label in labels])
    mean = np.average(features, axis=0, weights=weights)
    centered = features - mean
    target = labels * 2 - 1
    gram = centered.T @ (centered * weights[:, None])
    normal = np.linalg.solve(gram + alpha*np.eye(features.shape[1]),
                             centered.T @ (weights * target))
    if np.linalg.norm(normal) <= 1e-12:
        raise ValueError('The probe has no usable concept direction')
    return {'normal': normal, 'mean': mean, 'alpha': alpha}


def predict(features, probe):
    """Predict target state when the signed decision score is positive."""
    return ((features - probe['mean']) @ probe['normal'] > 0).astype(int)


def balanced_accuracy(labels, predictions):
    """Average recall over both classes; both must exist in evaluation data."""
    labels = np.asarray(labels)
    if set(labels) != {0, 1}:
        raise ValueError('Evaluation requires both classes')
    return float(np.mean([np.mean(predictions[labels == label] == label) for label in [0, 1]]))
