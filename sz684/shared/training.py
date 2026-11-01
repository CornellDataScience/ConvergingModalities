"""Training: centered ridge text-to-image translator shared across experiments."""

import torch


def fit_ridge(x, y, alpha=1.0):
    # Solve min_W ||(x - mean(x)) W - (y - mean(y))||² + alpha ||W||².
    # The affine intercept cancels for differences. To predict an endpoint,
    # use (x_new - mean(x)) @ W + mean(y). No encoder weights are updated.
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    return torch.linalg.solve(
        xc.T @ xc + alpha * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ yc
    )
