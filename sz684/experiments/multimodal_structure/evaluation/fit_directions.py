"""Fit train-only maps/probes and evaluate each relation on untouched families."""
from pathlib import Path
from itertools import permutations
import numpy as np
from shared.io import write_json
from ..data.prepare import load_manifest
from ..extraction.cache import load_features
from ..training.preprocessing import fit_pca, transform
from ..training.alignment import fit_procrustes, map_direction
from ..training.probes import fit_probe, predict, balanced_accuracy
from .evaluate import concept_centroids
from .directions import evaluate_transfer
from .retrieval import retrieve, select_scale


def json_arrays(value):
    """Convert fitted NumPy arrays into readable, reproducible JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_arrays(item) for item in value]
    return value


def relation_rows(features, ids, records, relation, split, families=None):
    """Select one relation and collapse repeated content/source observations."""
    index = {row['sample_id']: row for row in records}
    selected, rows, seen = [], [], set()
    for feature, sample_id in zip(features, ids):
        row = index[sample_id]
        if row.get('relation') != relation or row['split'] != split:
            continue
        if families is not None and row['family_id'] not in families:
            continue
        key = (row['concept_id'], row.get('sha256', row['source_id']))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        selected.append(feature)
    if not selected or {str(row['label']) for row in rows} != {'0','1'}:
        raise ValueError(f'{relation}: missing both endpoint labels in {split}')
    return np.stack(selected), rows, np.array([int(row['label']) for row in rows])


def fit_directions(manifest_path, cache_paths, output, dimensions=8, seed=42):
    """Evaluate both map directions, with separate anchor/probe training families.

    Hyperparameters: fixed PCA width; ridge alpha selected on validation balanced
    accuracy; retrieval scale selected on validation AP. Test data never select
    a parameter. Sparse relations are recorded as skipped, not silently omitted.
    """
    manifest = load_manifest(manifest_path)
    records = manifest['records']
    families = sorted({row['family_id'] for row in records if row['split'] == 'train'})
    if len(families) < 4:
        raise ValueError('Need at least four training families for separate anchors/probes')
    order = np.random.default_rng(seed).permutation(families)
    anchor_families = set(order[:len(order)//2])
    probe_families = set(order[len(order)//2:])
    models = {}
    identities = {}
    for name, path in cache_paths.items():
        raw, ids, identity = load_features(path)
        if identity['dataset_hash'] != manifest['sample_hash']:
            raise ValueError('Cache/dataset mismatch')
        models[name] = (raw, ids)
        identities[name] = identity
    relations = sorted({row['relation'] for row in records if row.get('relation')})
    results, fits, skipped = [], {}, []
    for source, target in permutations(models, 2):
        try:
            anchor_centroids = {}
            for name in [source, target]:
                raw, ids = models[name]
                anchor_records = [row for row in records if row['family_id'] in anchor_families]
                anchor_ids = {row['sample_id'] for row in anchor_records}
                selected = [i for i, sample_id in enumerate(ids) if sample_id in anchor_ids]
                anchor_centroids[name] = concept_centroids(raw[selected], [ids[i] for i in selected], anchor_records, 'train')
            concepts = sorted(set(anchor_centroids[source]) & set(anchor_centroids[target]))
            a = np.stack([anchor_centroids[source][concept] for concept in concepts])
            b = np.stack([anchor_centroids[target][concept] for concept in concepts])
            tx, ty = fit_pca(a, dimensions), fit_pca(b, dimensions)
            alignment = fit_procrustes(transform(a, tx), transform(b, ty))
            pair_name = source+'__to__'+target
            fits[pair_name] = {'source_transform': tx, 'target_transform': ty,
                              'alignment': alignment, 'anchor_concepts': concepts, 'relations': {}}
        except ValueError as error:
            skipped.append({'source': source, 'target': target, 'reason': str(error)})
            continue
        for relation in relations:
            try:
                probes, data = {}, {}
                for name, fitted in [(source, tx), (target, ty)]:
                    raw, ids = models[name]
                    data[name] = {}
                    for split in ['train','validation','test']:
                        x, rows, labels = relation_rows(raw, ids, records, relation, split,
                                                        probe_families if split == 'train' else None)
                        data[name][split] = (transform(x, fitted), rows, labels)
                    training, _, labels = data[name]['train']
                    validation, _, validation_labels = data[name]['validation']
                    candidates = [fit_probe(training, labels, alpha) for alpha in [0.1, 1, 10]]
                    scores = [balanced_accuracy(validation_labels, predict(validation, p)) for p in candidates]
                    probes[name] = candidates[int(np.argmax(scores))]
                test, rows, labels = data[target]['test']
                score = evaluate_transfer(probes[source], probes[target], alignment, test, labels)
                # CAV stepping is an operational retrieval test, not an assertion
                # that classifier normals equal mean endpoint displacement.
                direction = map_direction(probes[source]['normal'], alignment)
                direction = direction/np.linalg.norm(direction)
                validation, validation_rows, _ = data[target]['validation']
                scale = select_scale(validation, validation_rows, direction, [0, .25, .5, 1, 2, 4])
                rng = np.random.default_rng(seed)
                random = rng.normal(size=dimensions)
                random /= np.linalg.norm(random)
                score['retrieval'] = {name: retrieve(test, rows, vector, magnitude)
                                      for name, vector, magnitude in [('no_shift',direction,0),
                                          ('mapped_cav',direction,scale), ('reverse',-direction,scale),
                                          ('random',random,scale)]}
                score['scale'] = scale
                results.append(dict(source=source, target=target, relation=relation, **score))
                fits[pair_name]['relations'][relation] = {'source_probe': probes[source], 'target_probe': probes[target]}
            except ValueError as error:
                skipped.append({'source': source, 'target': target, 'relation': relation, 'reason': str(error)})
    result = {'status': 'EXPLORATORY fixed-fit transfer; no refit confidence intervals',
              'dataset_hash': manifest['sample_hash'], 'models': identities,
              'anchor_families': sorted(anchor_families), 'probe_families': sorted(probe_families),
              'dimensions': dimensions, 'seed': seed, 'results': results, 'skipped': skipped,
              'fits': json_arrays(fits)}
    write_json(Path(output), result)
    return result
