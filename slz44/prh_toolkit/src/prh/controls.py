"""Anisotropy controls: fit on training rows only, apply to any rows. Pure numpy.

Inputs are expected to be clipped and L2-normalized [N, D] feature matrices. Outputs are L2-normalized again,
except for `baseline`, which returns its input unchanged.
"""
import numpy as np

MODES = ('baseline', 'center', 'pca128', 'standardize', 'remove3', 'pca128_whiten')
DIAGNOSTICS = ('mean_cosine', 'effective_rank', 'top_dim_share', 'top_pc_share')


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def clip_normalize(x, threshold):
    """Clip each value to [-threshold, threshold], then L2-normalize (the baseline preprocessing)."""
    return l2_normalize(np.clip(x, -threshold, threshold))


def fit_control(train, mode, dimensions=128, top=3):
    """Fit one control on training rows. Returns a dict that apply_control understands."""
    if mode not in MODES:
        raise ValueError(f'unknown mode {mode!r}; choose from {MODES}')
    if mode == 'baseline':
        return dict(mode=mode, retained=None)
    mean = train.mean(0)
    centered = train - mean
    if mode == 'center':
        return dict(mode=mode, mean=mean, retained=None)
    if mode == 'standardize':
        return dict(mode=mode, mean=mean, scale=np.maximum(centered.std(0, ddof=1), 1e-8), retained=None)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    energy = singular ** 2
    if mode == 'pca128':
        return dict(mode=mode, mean=mean, basis=vh[:dimensions].T.copy(),
                    retained=float(energy[:dimensions].sum() / energy.sum()))
    if mode == 'remove3':
        return dict(mode=mode, mean=mean, basis=vh[:top].T.copy(),
                    retained=float(1 - energy[:top].sum() / energy.sum()))
    scale = np.maximum(singular[:dimensions] / np.sqrt(len(train) - 1), 1e-8)
    return dict(mode=mode, mean=mean, basis=vh[:dimensions].T.copy(), scale=scale,
                retained=float(energy[:dimensions].sum() / energy.sum()))


def apply_control(x, fitted):
    mode = fitted['mode']
    if mode == 'baseline':
        return x
    x = x - fitted['mean']
    if mode == 'standardize':
        x = x / fitted['scale']
    elif mode == 'remove3':
        x = x - (x @ fitted['basis']) @ fitted['basis'].T
    elif mode == 'pca128':
        x = x @ fitted['basis']
    elif mode == 'pca128_whiten':
        x = (x @ fitted['basis']) / fitted['scale']
    return l2_normalize(x)


def diagnostics(x):
    """Geometry of L2-normalized vectors [N, D]: mean pairwise cosine, effective rank, variance shares."""
    n = len(x)
    sims = x @ x.T
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = singular / singular.sum()
    variance = x.var(0, ddof=1)
    return dict(mean_cosine=float((sims.sum() - np.trace(sims)) / (n * (n - 1))),
                effective_rank=float(np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum())),
                top_dim_share=float(variance.max() / variance.sum()),
                top_pc_share=float(singular[0] ** 2 / (singular ** 2).sum()))
