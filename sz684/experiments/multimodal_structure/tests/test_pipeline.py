"""A tiny synthetic three-modality fixture verifies orchestration, not science."""
import json
import numpy as np
import pytest
from shared.io import digest, write_json
from ..data.prepare import validate_records
from ..data.split import assign_splits
from ..extraction.cache import save_features, load_features
from ..evaluation.evaluate import evaluate
from ..evaluation.fit_directions import fit_directions
from ..reporting.report import report


def test_source_leakage_is_rejected():
    rows = [dict(sample_id=str(i),concept_id=str(i), family_id=str(i),source_id='same',
                 modality='text',text='example',split=split)
            for i, split in enumerate(['train','test'])]
    with pytest.raises(ValueError, match='source_id crosses'):
        validate_records(rows)


def test_split_joins_related_sources():
    rows = [dict(family_id=str(i), source_id=str(i)) for i in range(8)]
    rows.append(dict(family_id='8', source_id='0'))
    result = assign_splits(rows)
    assert result[0]['split'] == result[-1]['split']


def test_cache_reordering_is_rejected(tmp_path):
    path = tmp_path/'cache.npz'
    identity = {'sample_ids':['b','a']}
    save_features(path, np.eye(2), ['a','b'], identity)
    with pytest.raises(ValueError, match='rows'):
        load_features(path)


def test_three_modality_pipeline(tmp_path):
    rng = np.random.default_rng(24)
    records, caches = [], {}
    base = rng.normal(size=(20, 3))
    axis = np.array([2., 0, 0])
    for modality in ['text','vision','audio']:
        rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
        values, ids = [], []
        for family in range(20):
            split = 'train' if family < 12 else 'validation' if family < 16 else 'test'
            for label in [0, 1]:
                for example in range(2):
                    sample_id = f'{modality}_{family}_{label}_{example}'
                    records.append(dict(sample_id=sample_id, concept_id=f'{family}_{label}',
                        family_id=str(family), source_id=f'{family}_{label}_{example}',
                        modality=modality, split=split, text='test',path='synthetic-not-media',
                        relation='axis', label=str(label)))
                    ids.append(sample_id)
                    values.append((base[family]+label*axis)@rotation)
        caches[modality] = (np.stack(values), ids)
    manifest = {'records':records, 'sample_hash':digest(records), 'status':'SYNTHETIC SOFTWARE TEST'}
    path = tmp_path/'manifest.json'
    write_json(path, manifest)
    paths = {}
    for name, (values, ids) in caches.items():
        paths[name] = tmp_path/(name+'.npz')
        save_features(paths[name], values, ids, {'sample_ids':ids,'dataset_hash':manifest['sample_hash'],
                                               'model':{'modality':name}})
    result = evaluate(path, paths, tmp_path/'scores.json', k=2, draws=5)
    assert len(result['pairs']) == 3
    assert all(pair['metrics']['cka']['score'] > .999 for pair in result['pairs'])
    transfer = fit_directions(path, paths, tmp_path/'directions.json', dimensions=3)
    assert len(transfer['results']) == 6
    assert not transfer['skipped']
    assert all(row['cav_cosine'] > .999 for row in transfer['results'])
    report(tmp_path/'scores.json', tmp_path/'report.md')
    assert 'SYNTHETIC' in (tmp_path/'report.md').read_text()
