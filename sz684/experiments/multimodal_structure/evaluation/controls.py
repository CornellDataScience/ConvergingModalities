"""Correspondence shuffles preserve each model's entire internal geometry."""
import numpy as np
from .rsa import rsa
from .cka import linear_cka
from .neighbors import neighbor_scores


def metrics(left, right, k):
    """Calculate the prespecified geometry summaries for a model pair."""
    graph = neighbor_scores(left, right, k)
    return {'rsa': rsa(left, right), 'cka': linear_cka(left, right),
            'jaccard': float(graph['jaccard'].mean()),
            'overlap': float(graph['overlap'].mean())}


def permutation_test(left, right, k, draws=199, seed=0):
    """Exploratory one-sided correspondence test; rows are whole concepts.

    Concepts must be exchangeable for these p-values. This is not an adjusted
    confirmatory test or a substitute for source/family cluster uncertainty.
    """
    if draws < 1:
        raise ValueError('At least one permutation is required')
    observed = metrics(left, right, k)
    rng = np.random.default_rng(seed)
    null = {name: [] for name in observed}
    for _ in range(draws):
        shuffled = metrics(left, right[rng.permutation(len(right))], k)
        for name, value in shuffled.items():
            null[name].append(value)
    return {name: {'score': value, 'null_mean': float(np.mean(null[name])),
                   'p_unadjusted': (1 + sum(x >= value for x in null[name])) / (draws + 1),
                   'null': null[name]} for name, value in observed.items()}
