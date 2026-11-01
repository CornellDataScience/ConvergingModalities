"""Assign whole families while linking any families that share a source."""
import numpy as np


def assign_splits(records, seed=42):
    """Return copied records with deterministic 60/20/20 group assignments.

    Existing splits are replaced. This helper never chooses a split based on
    model scores. Shared source IDs join families before groups are assigned.
    """
    families = sorted({row['family_id'] for row in records})
    parent = {family: family for family in families}

    def root(family):
        while parent[family] != family:
            family = parent[family]
        return family

    sources = {}
    for row in records:
        family = row['family_id']
        if row['source_id'] in sources:
            parent[root(family)] = root(sources[row['source_id']])
        sources[row['source_id']] = family
    groups = sorted({root(family) for family in families})
    if len(groups) < 5:
        raise ValueError('Need at least five independent family/source groups')
    order = np.random.default_rng(seed).permutation(groups)
    train_end = max(1, int(len(order)*0.6))
    validation_end = max(train_end+1, int(len(order)*0.8))
    assignment = {group: ('train' if i < train_end else 'validation' if i < validation_end else 'test')
                  for i, group in enumerate(order)}
    return [dict(row, split=assignment[root(row['family_id'])]) for row in records]
