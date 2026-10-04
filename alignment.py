"""CPU float32 reference for the authors' mutual-neighbor measurement."""
import numpy as np
import torch
import torch.nn.functional as F


def prepare(features, q=0.95, mode="upstream_default"):
    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if x.ndim != 3 or not torch.isfinite(x).all():
        raise ValueError("Expected finite [samples, layers, dimensions] features")
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    if q < 1:
        if mode == "upstream_default":
            threshold = torch.quantile(x.abs().flatten(1), q, dim=1).mean()
        elif mode == "upstream_precise":
            flat = x.abs().flatten()
            threshold = flat.kthvalue(int(q * flat.numel()) + 1).values
        else:
            raise ValueError(f"Unknown clipping mode: {mode}")
        x = x.clamp(-threshold, threshold)
    return F.normalize(x, dim=-1)


def neighbors(x, k):
    if x.ndim != 2 or not 0 < k < len(x):
        raise ValueError("Need [N,D] features and 0 < k < N")
    similarities = x @ x.T
    similarities.fill_diagonal_(-float("inf"))
    return similarities.argsort(dim=1, descending=True)[:, :k].numpy()


def overlap(a, b):
    if a.shape != b.shape:
        raise ValueError("Neighbor arrays must have equal shapes")
    return (a[:, :, None] == b[:, None, :]).any(axis=2).mean().item()


def score_layers(a, b, k=10, permutations=100, seed=0):
    if len(a) != len(b):
        raise ValueError("Sample counts differ")
    na = [neighbors(a[:, i], k) for i in range(a.shape[1])]
    nb = [neighbors(b[:, i], k) for i in range(b.shape[1])]
    def matrix(right_neighbors):
        # Membership lookup is O(N*k), rather than O(N*k*k), per layer pair.
        rows = np.arange(len(a))[:, None]
        values = np.empty((len(na), len(right_neighbors)))
        for j, right in enumerate(right_neighbors):
            membership = np.zeros((len(a), len(a)), dtype=bool)
            membership[rows, right] = True
            for i, left in enumerate(na):
                values[i, j] = membership[rows, left].mean()
        return values
    scores = matrix(nb)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        p = rng.permutation(len(b))
        inverse = np.argsort(p)
        shuffled = [inverse[y[p]] for y in nb]
        null.append(float(matrix(shuffled).max()))
    best = float(scores.max())
    return {"scores": scores.tolist(), "best_score": best,
            "best_layers": list(map(int, np.unravel_index(scores.argmax(), scores.shape))),
            "permuted_maxima": null,
            "permutation_p": (1 + sum(v >= best for v in null)) / (1 + len(null)) if null else None,
            "single_pair_random_expectation": k / (len(a) - 1)}
