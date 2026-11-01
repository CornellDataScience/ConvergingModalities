"""The fit/lock boundary must prevent test-driven parameter changes."""
import json
from pathlib import Path
import numpy as np
import pytest
from shared.io import digest,write_json
from ..data.expanded import broad_family
from ..extraction.cache import save_features,load_features
from ..training.build_fits import build_fits
from ..evaluation.locked import lock_evaluation,score_locked


def tiny_study(folder):
    """Return real-format synthetic data with both axes in all three splits."""
    rng=np.random.default_rng(51)
    family_splits=[('instrument_percussion','train'),('canines','train'),
                   ('instrument_brass','validation'),('human_vocal_and_body','validation'),
                   ('instrument_strings','test'),('felines','test')]
    rows=[];matrices={};ids={}
    base=rng.normal(size=(24,4))
    for model,modality in [('image','vision'),('sound','audio')]:
        rotation,_=np.linalg.qr(rng.normal(size=(4,4)))
        values=[];samples=[]
        for group,(family,split) in enumerate(family_splits):
            for concept in range(4):
                index=group*4+concept
                for example in range(4):
                    sample=f'{model}_{index}_{example}'
                    rows.append(dict(sample_id=sample,concept_id=str(index),family_id=family,
                                     source_id=f'{index}_{example}',modality=modality,split=split,path='synthetic'))
                    values.append(base[index]@rotation+example*.001)
                    samples.append(sample)
        matrices[model]=np.stack(values);ids[model]=samples
    manifest={'records':rows,'sample_hash':digest(rows),'status':'SYNTHETIC SOFTWARE TEST'}
    path=folder/'manifest.json';write_json(path,manifest)
    caches={}
    for name,values in matrices.items():
        caches[name]=folder/(name+'.npz')
        identity={'sample_ids':ids[name],'dataset_hash':manifest['sample_hash'],
                  'model':{'modality':'vision' if name=='image' else 'audio'}}
        save_features(caches[name],values,ids[name],identity)
    return path,caches


def test_taxonomy_uses_whole_words():
    assert broad_family('cat growling')=='felines'
    assert broad_family('coyote howling')=='canines'
    assert broad_family('people crowd')=='human_vocal_and_body'
    assert broad_family('playing bassoon')=='instrument_woodwind'
    assert broad_family('playing bass guitar')=='instrument_strings'


def test_test_features_do_not_choose_fitted_parameters(tmp_path):
    manifest,caches=tiny_study(tmp_path)
    first=build_fits(manifest,caches,tmp_path/'first.json',dimensions=(2,3))
    records={row['sample_id']:row for row in json.loads(manifest.read_text())['records']}
    for path in caches.values():
        values,ids,identity=load_features(path)
        for i,sample in enumerate(ids):
            if records[sample]['split']=='test': values[i]*=1000
        save_features(path,values,ids,identity)
    second=build_fits(manifest,caches,tmp_path/'second.json',dimensions=(2,3))
    for key in ['transforms','probes','pairs']:
        assert first[key]==second[key]


def test_lock_and_single_test_execution(tmp_path):
    manifest,caches=tiny_study(tmp_path)
    fitted=tmp_path/'fits.json';lock=tmp_path/'lock.json'
    build_fits(manifest,caches,fitted,dimensions=(2,3))
    lock_evaluation(manifest,caches,fitted,lock,k=2,draws=3)
    result=score_locked(manifest,caches,fitted,lock,tmp_path/'test.json')
    assert len(result['category_axis_results'])==4
    assert len(result['test_concepts'])==8
    with pytest.raises(ValueError,match='already exists'):
        score_locked(manifest,caches,fitted,lock,tmp_path/'test.json')
    # Tampering with the feature matrix must also stop an otherwise new run.
    path=next(iter(caches.values()))
    values,ids,identity=load_features(path);values[0,0]+=1
    save_features(path,values,ids,identity)
    with pytest.raises(ValueError,match='features changed'):
        score_locked(manifest,caches,fitted,lock,tmp_path/'tampered.json')


def test_changed_features_cannot_be_locked_as_old_fits(tmp_path):
    manifest,caches=tiny_study(tmp_path)
    fitted=tmp_path/'fits.json'
    build_fits(manifest,caches,fitted,dimensions=(2,))
    path=next(iter(caches.values()))
    values,ids,identity=load_features(path);values[0,0]+=1
    save_features(path,values,ids,identity)
    with pytest.raises(ValueError,match='changed after fitting'):
        lock_evaluation(manifest,caches,fitted,tmp_path/'lock.json')
