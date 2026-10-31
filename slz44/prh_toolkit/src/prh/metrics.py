"""Alignment metrics on paired feature matrices [N, D] whose rows are the same samples. Pure numpy; POT optional."""
import numpy as np


def neighbors(x, k):
    """Indices of each row's k nearest neighbors by inner product (rows should be L2-normalized)."""
    if x.ndim != 2 or not 0 < k < len(x):
        raise ValueError('Need [N,D] features and 0 < k < N')
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1, kind='stable')[:, :k]


def overlap(a, b):
    """Mean fraction of neighbors shared between two neighbor arrays of equal shape."""
    if a.shape != b.shape:
        raise ValueError('Neighbor arrays must have equal shapes')
    return float((a[:, :, None] == b[:, None, :]).any(axis=2).mean())


def mutual_knn(a, b, k):
    """Mutual k-NN alignment: average overlap of each sample's k neighbors in the two spaces."""
    return overlap(neighbors(a, k), neighbors(b, k))


def centered_gram(x):
    k = x @ x.T
    r = k.mean(0)
    return k - r[None, :] - r[:, None] + k.mean()


def cka_gram(k, l):
    return float((k * l).sum() / np.sqrt((k * k).sum() * (l * l).sum()))


def linear_cka(a, b):
    """Linear CKA (Kornblith et al. 2019). Centers features, so a constant offset is ignored."""
    return cka_gram(centered_gram(a), centered_gram(b))


def normalized_distance(x):
    """Cosine-distance matrix scaled so the mean off-diagonal distance is 1."""
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    d = np.clip(1 - xn @ xn.T, 0, None)
    np.fill_diagonal(d, 0)
    n = len(d)
    return d / (d.sum() / (n * (n - 1)))


def gw_cost(c1, c2, perm=None):
    """Mean squared difference between distance matrices under a pairing (identity by default)."""
    other = c2 if perm is None else c2[np.ix_(perm, perm)]
    return float(((c1 - other) ** 2).mean())


def gw_match(c1, c2, max_iter=200):
    """Blind matching with Gromov-Wasserstein optimal transport (needs POT).

    Returns top-1 / top-5 accuracy of the coupling against the true pairing (chance 1/N, 5/N).
    """
    import ot
    n = len(c1)
    p = np.full(n, 1 / n)
    coupling, log = ot.gromov.gromov_wasserstein(c1, c2, p, p, 'square_loss', log=True, max_iter=max_iter)
    order = np.argsort(-coupling, axis=1)
    rows = np.arange(n)
    return dict(top1=float((order[:, 0] == rows).mean()),
                top5=float((order[:, :5] == rows[:, None]).any(1).mean()),
                gw_distance=float(log['gw_dist']))
