"""Data transformations: feature-cache loading, train-fitted PCA, gallery masks."""

import json
import numpy as np
import torch
import torch.nn.functional as F


def fit_preprocessing(raw, q95, rows, dimensions):
    # Fit clipping and PCA only on map-fitting rows; the closure then applies
    # those fixed parameters to every split, with endpoint renormalization.
    threshold = q95[rows].mean()
    normalized = F.normalize(raw.clamp(-threshold, threshold), dim=-1)
    mean = normalized[rows].mean(0)
    _, singular, vh = torch.linalg.svd(normalized[rows] - mean, full_matrices=False)
    if len(singular) < dimensions or singular[dimensions - 1] <= singular[0] * 1e-8:
        raise ValueError("Insufficient effective rank for declared PCA dimension")
    basis = vh[:dimensions].T.contiguous()

    def transform(x):
        return F.normalize(
            (F.normalize(x.clamp(-threshold, threshold), dim=-1) - mean) @ basis, dim=-1
        )

    return transform, dict(
        threshold=float(threshold),
        retained_variance=float(
            singular[:dimensions].square().sum() / singular.square().sum()
        ),
        smallest_retained_relative_singular=float(
            singular[dimensions - 1] / singular[0]
        ),
    )


def load_selected(run, modality):
    folder = run / ("features_" + modality)
    meta = json.loads((folder / "metadata.json").read_text())
    if not meta["complete"]:
        raise ValueError("Incomplete feature cache")
    arrays = []
    quantiles = []
    rows = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as a:
            arrays.append(a["features"])
            quantiles.append(a["q95"])
            rows.extend(a["rows"].tolist())
    if rows != list(range(len(rows))):
        raise ValueError("Missing or repeated feature rows")
    return (
        torch.tensor(np.concatenate(arrays), dtype=torch.float64),
        torch.tensor(np.concatenate(quantiles), dtype=torch.float64),
        meta,
    )


def bank_allowed(categories, instances):
    # Shared by color and natural-state analyses. Query order: noun, instance;
    # gallery order: noun, instance, source/target state. Exclude the source
    # index and cyclic target index in every noun to balance gallery counts.
    # Natural-state indices do not identify matched before/after photographs.
    q = np.tile(np.arange(instances), categories)
    bank_instance = np.repeat(np.tile(np.arange(instances), categories), 2)
    bank_color = np.tile([0, 1], categories * instances)
    return ~(
        ((bank_color[None, :] == 0) & (bank_instance[None, :] == q[:, None]))
        | (
            (bank_color[None, :] == 1)
            & (bank_instance[None, :] == ((q + 1) % instances)[:, None])
        )
    )
