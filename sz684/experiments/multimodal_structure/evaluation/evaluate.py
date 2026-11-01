"""Compare all supplied model caches on exactly the same held-out concepts."""
from itertools import combinations
from pathlib import Path
import numpy as np
from shared.io import write_json
from ..data.prepare import load_manifest
from ..extraction.cache import load_features
from .controls import permutation_test


def concept_centroids(features, sample_ids, records, split):
    """Average within source first so repeated frames do not get extra weight."""
    index = {row['sample_id']: row for row in records}
    groups = {}
    for feature, sample_id in zip(features, sample_ids):
        row = index[sample_id]
        if split == 'all' or row['split'] == split:
            key = (row['concept_id'], row['source_id'])
            groups.setdefault(key, []).append(feature)
    sources = {}
    for (concept, source), values in groups.items():
        sources.setdefault(concept, []).append(np.mean(values, axis=0))
    return {concept: np.mean(values, axis=0) for concept, values in sources.items()}


def evaluate(manifest_path, cache_paths, output, split='test', k=3, draws=199, seed=42):
    """Write exploratory scores and whole-concept correspondence nulls."""
    manifest = load_manifest(manifest_path)
    models, identities = {}, {}
    for name, path in cache_paths.items():
        features, ids, identity = load_features(path)
        if identity['dataset_hash'] != manifest['sample_hash']:
            raise ValueError('Feature cache belongs to a different dataset')
        models[name] = concept_centroids(features, ids, manifest['records'], split)
        identities[name] = identity
    if len(models) < 2:
        raise ValueError('Supply at least two models')
    concepts = sorted(set.intersection(*[set(values) for values in models.values()]))
    if len(concepts) <= max(k, 2):
        raise ValueError('Too few concepts shared by all models in this split')
    pairs = []
    for left, right in combinations(models, 2):
        a = np.stack([models[left][concept] for concept in concepts])
        b = np.stack([models[right][concept] for concept in concepts])
        pairs.append({'left': left, 'right': right,
                      'metrics': permutation_test(a, b, k, draws, seed)})
    result = {'status': 'EXPLORATORY; unadjusted exchangeable-concept permutation tests',
              'dataset_status': manifest.get('status'), 'dataset_hash': manifest['sample_hash'],
              'split': split, 'concepts': concepts, 'k': k, 'draws': draws, 'seed': seed,
              'models': identities, 'pairs': pairs}
    write_json(Path(output), result)
    return result
