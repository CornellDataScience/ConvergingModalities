"""Training: fit the shared state-transfer bridge on paired fitting rows only."""

import torch
from shared.training import fit_ridge


def fit_bridge(x, y, alpha):
    """Return ridge W, endpoint means, and the orthogonal comparison map.

    Both encoders are frozen. The caller supplies only fitting-noun rows after
    train-fitted preprocessing; validation and test embeddings must stay out.
    """
    w = fit_ridge(x, y, alpha)
    xm, ym = x.mean(0), y.mean(0)
    u, _, vh = torch.linalg.svd((x - xm).T @ (y - ym), full_matrices=False)
    return w, xm, ym, u @ vh
