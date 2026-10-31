"""Shuffled-pairing null models: how much of a score would a random pairing of the same samples give?

Every function re-pairs the rows of the second space and recomputes the same statistic, so null and observed
share the same preprocessing, neighborhoods and sample size.
"""
import numpy as np
from .metrics import cka_gram, centered_gram, gw_cost, neighbors, normalized_distance, overlap


def knn_null(a, b, k, permutations=1000, seed=0):
    """Mutual k-NN with a shuffled-pairing null. `adjusted` rescales the excess by the room above the null."""
    left, right = neighbors(a, k), neighbors(b, k)
    score = overlap(left, right)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        p = rng.permutation(len(right))
        null.append(overlap(left, np.argsort(p)[right[p]]))
    mu = float(np.mean(null))
    return dict(score=score, null_mean=mu, null_q95=float(np.quantile(null, .95)), excess=score - mu,
                adjusted=(score - mu) / (1 - mu),
                p=(1 + sum(v >= score for v in null)) / (1 + permutations), null=null)


def cka_null(a, b, permutations=200, seed=0):
    k, l = centered_gram(a), centered_gram(b)
    score = cka_gram(k, l)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        p = rng.permutation(len(l))
        null.append(cka_gram(k, l[np.ix_(p, p)]))
    return dict(score=score, null_mean=float(np.mean(null)), excess=score - float(np.mean(null)), null=null)


def gw_cost_null(a, b, permutations=200, seed=0):
    """Structure discrepancy at the true pairing vs shuffled pairings. reduction > 0 means aligned."""
    c1, c2 = normalized_distance(a), normalized_distance(b)
    observed = gw_cost(c1, c2)
    rng = np.random.default_rng(seed)
    null = [gw_cost(c1, c2, rng.permutation(len(c1))) for _ in range(permutations)]
    mu = float(np.mean(null))
    return dict(cost=observed, null_mean=mu, reduction=(mu - observed) / mu, null=null)
