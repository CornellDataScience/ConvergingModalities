"""Data transformations: train-fitted centering/PCA for the WIT studies."""

import torch
import torch.nn.functional as F


def fit_transform(train, mode, dimensions=128):
    if mode == "baseline":
        return None, None, None
    mean = train.mean(0)
    if mode == "center":
        return mean, None, None
    if mode != "pca128":
        raise ValueError(mode)
    _, singular, vh = torch.linalg.svd(train - mean, full_matrices=False)
    basis = vh[:dimensions].T.contiguous()
    retained = float(singular[:dimensions].square().sum() / singular.square().sum())
    return mean, basis, retained


def apply_transform(x, fitted):
    mean, basis, _ = fitted
    if mean is None:
        return x
    x = x - mean
    if basis is not None:
        x = x @ basis
    return F.normalize(x, dim=-1)
