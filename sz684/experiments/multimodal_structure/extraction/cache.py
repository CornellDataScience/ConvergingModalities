"""Explicit cache identity and atomic writes prevent silent model/row mixups."""
import json
from pathlib import Path
import numpy as np
from shared.io import digest, write_json


def cache_identity(manifest, model, records):
    """All extraction settings belong in model, including revision and pooling."""
    return {'dataset_hash': manifest['sample_hash'], 'model': model,
            'sample_ids': [row['sample_id'] for row in records], 'extractor_version': 1}


def save_features(path, features, sample_ids, identity):
    """Write a complete [examples, coordinates] cache with embedded metadata."""
    path = Path(path)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(sample_ids) or not np.isfinite(features).all():
        raise ValueError('Invalid feature matrix')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('Duplicate cached sample IDs')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, features=features, sample_ids=np.array(sample_ids),
                 identity=json.dumps(identity, sort_keys=True))
    temporary.replace(path)


def load_features(path, expected=None):
    """Load without pickle and reject reordered rows or a different identity."""
    with np.load(path, allow_pickle=False) as cache:
        features = cache['features'].astype(np.float64)
        sample_ids = cache['sample_ids'].tolist()
        identity = json.loads(str(cache['identity']))
    if features.ndim != 2 or len(features) != len(sample_ids) or not np.isfinite(features).all():
        raise ValueError('Invalid cached features')
    if len(set(sample_ids)) != len(sample_ids) or sample_ids != identity['sample_ids']:
        raise ValueError('Feature rows differ from recorded identities')
    if expected is not None and identity != expected:
        raise ValueError('Cache identity mismatch; use another output directory')
    return features, sample_ids, identity
