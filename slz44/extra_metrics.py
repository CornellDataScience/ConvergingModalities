"""Alignment metrics beyond k-NN: linear CKA and Gromov-Wasserstein (numpy; POT optional).

All scores are computed on paired feature matrices [N, D] whose rows are the same samples.
"""
import numpy as np


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def centered_gram(x):
    k = x @ x.T
    r = k.mean(0)
    return k - r[None, :] - r[:, None] + k.mean()


def cka_gram(k, l):
    return float((k * l).sum() / np.sqrt((k * k).sum() * (l * l).sum()))


def linear_cka(a, b):
    """Linear CKA (Kornblith et al. 2019). Centers features, so a constant offset is ignored."""
    return cka_gram(centered_gram(a), centered_gram(b))


def cka_with_null(a, b, permutations=200, seed=0):
    k, l = centered_gram(a), centered_gram(b)
    score = cka_gram(k, l)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        p = rng.permutation(len(l))
        null.append(cka_gram(k, l[np.ix_(p, p)]))
    return dict(score=score, null_mean=float(np.mean(null)), excess=score - float(np.mean(null)), null=null)


def normalized_distance(x):
    """Cosine-distance matrix scaled so the mean off-diagonal distance is 1."""
    xn = unit(x)
    d = np.clip(1 - xn @ xn.T, 0, None)
    np.fill_diagonal(d, 0)
    n = len(d)
    return d / (d.sum() / (n * (n - 1)))


def gw_cost(c1, c2, perm=None):
    """Mean squared difference between distance matrices under a pairing (identity by default)."""
    other = c2 if perm is None else c2[np.ix_(perm, perm)]
    return float(((c1 - other) ** 2).mean())


def gw_cost_with_null(c1, c2, permutations=200, seed=0):
    """Structure discrepancy at the true pairing vs shuffled pairings. reduction > 0 means aligned."""
    observed = gw_cost(c1, c2)
    rng = np.random.default_rng(seed)
    null = [gw_cost(c1, c2, rng.permutation(len(c1))) for _ in range(permutations)]
    mu = float(np.mean(null))
    return dict(cost=observed, null_mean=mu, reduction=(mu - observed) / mu, null=null)


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
