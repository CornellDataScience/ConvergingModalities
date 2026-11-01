"""Week 8: score saved fits once, after locking settings and software identity."""
import csv
import json
from pathlib import Path
import numpy as np
from shared.io import write_json, digest
from ..provenance import file_hash,software_hashes
from ..data.axes import AXES,axis_label
from ..training.build_fits import load_study,select_axis
from ..training.preprocessing import transform
from ..training.probes import predict,balanced_accuracy
from ..training.alignment import map_endpoints
from .rsa import distance_matrix
from .neighbors import neighbors,neighbor_scores
from .evaluate import concept_centroids
from .directions import evaluate_transfer
from .evaluate import evaluate



def lock_evaluation(manifest_path, cache_paths, fits_path, output, k=3, draws=999):
    """Create an immutable prospective evaluation lock; never overwrite one."""
    output=Path(output)
    if output.exists(): raise ValueError('Evaluation is already locked')
    manifest,_,_=load_study(manifest_path,cache_paths)
    fits=json.loads(Path(fits_path).read_text())
    if fits['dataset_hash']!=manifest['sample_hash'] or fits['test_scores_computed']:
        raise ValueError('Invalid train/validation-only fit artifact')
    if fits['feature_sha256']!={name:file_hash(path) for name,path in cache_paths.items()}:
        raise ValueError('Feature files changed after fitting')
    if fits['software_hashes']!=software_hashes():
        raise ValueError('Software changed after fitting; rebuild fits before locking')
    root=Path(__file__).resolve().parents[1]
    protocol_hashes={str(path.relative_to(root)):file_hash(path) for path in [root/'configs/week8.json',root/'docs/WEEK8_PROTOCOL.md']}
    lock={'protocol_hashes':protocol_hashes,'dataset_hash':manifest['sample_hash'],'fits_sha256':file_hash(fits_path),
          'cache_sha256':{name:file_hash(path) for name,path in cache_paths.items()},
          'software_hashes':software_hashes(),'k':k,'draws':draws,'axes':fits['axes'],
          'status':'Prospective released-label evaluation; no new independent human verification',
          'primary_comparisons':['text-vision','audio-vision'],'secondary_comparisons':['text-audio','within-modality'],
          'interpretation':'Category-axis CAV transfer; not identity-preserving audiovisual state change'}
    write_json(output,lock)
    return lock


def array_dict(values):
    """Restore only numeric parameter arrays, leaving metadata readable."""
    return {key:np.array(value) if isinstance(value,list) else value for key,value in values.items()}


def score_locked(manifest_path, cache_paths, fits_path, lock_path, output):
    """Fail closed if data, fits, features or software changed after the lock."""
    output=Path(output)
    if output.exists(): raise ValueError('Test output already exists; preserve the first evaluation')
    lock=json.loads(Path(lock_path).read_text())
    manifest,data,identities=load_study(manifest_path,cache_paths)
    if lock['dataset_hash']!=manifest['sample_hash'] or lock['fits_sha256']!=file_hash(fits_path):
        raise ValueError('Dataset/fits changed after lock')
    if lock['cache_sha256']!={name:file_hash(path) for name,path in cache_paths.items()}:
        raise ValueError('Model features changed after lock')
    if lock['software_hashes']!=software_hashes(): raise ValueError('Software changed after lock')
    root=Path(__file__).resolve().parents[1]
    if any(file_hash(root/path)!=value for path,value in lock['protocol_hashes'].items()):
        raise ValueError('Protocol changed after lock')
    fits=json.loads(Path(fits_path).read_text())
    index={row['sample_id']:row for row in manifest['records']}
    results=[]
    predictions=[]
    for pair in fits['pairs']:
        source,target,key=pair['source'],pair['target'],str(pair['dimensions'])
        features,ids=data[target]
        projected=transform(features,array_dict(fits['transforms'][target][key]))
        for axis in fits['axes']:
            test,labels=select_axis(projected,ids,index,axis,'test')
            source_probe=array_dict(fits['probes'][source][key][axis])
            target_probe=array_dict(fits['probes'][target][key][axis])
            alignment=array_dict(pair['alignment'])
            score=evaluate_transfer(source_probe,target_probe,alignment,test,labels)
            shuffled=evaluate_transfer(source_probe,target_probe,array_dict(pair['shuffled_alignment']),test,labels)
            score['shuffled_anchor_accuracy']=shuffled['transfer_balanced_accuracy']
            reversed_probe=dict(source_probe,normal=-source_probe['normal'])
            reversed_score=evaluate_transfer(reversed_probe,target_probe,alignment,test,labels)
            score['reverse_accuracy']=reversed_score['transfer_balanced_accuracy']
            moved_probe={'normal':source_probe['normal'] @ alignment['rotation'],
                         'mean':map_endpoints(source_probe['mean'][None,:],alignment)[0]}
            predicted=predict(test,moved_probe)
            target_predicted=predict(test,target_probe)
            selected_ids=[sample for sample in ids if index[sample]['split']=='test'
                          and axis_label(axis,index[sample]['family_id']) is not None]
            for sample,label,guess,target_guess in zip(selected_ids,labels,predicted,target_predicted):
                row=index[sample]
                predictions.append(dict(source_model=source,target_model=target,axis=axis,sample_id=sample,
                    concept_id=row['concept_id'],family_id=row['family_id'],source_id=row['source_id'],
                    label=int(label),transferred_prediction=int(guess),target_prediction=int(target_guess)))
            results.append(dict(source=source,target=target,axis=axis,dimensions=int(key),
                                examples=len(labels),positive_examples=int(labels.sum()),**score))
    # Geometry does not use the fitted map or projected coordinates.
    geometry_path=output.with_name(output.stem+'_geometry.json')
    if geometry_path.exists(): raise ValueError('Geometry output exists; use a new evaluation run')
    geometry=evaluate(manifest_path,cache_paths,geometry_path,'test',lock['k'],lock['draws'],fits['seed'])
    details={'concepts':geometry['concepts'],'models':{},'pairs':[]}
    matrices={}
    for name,(features,ids) in data.items():
        centroids=concept_centroids(features,ids,manifest['records'],'test')
        matrices[name]=np.stack([centroids[concept] for concept in geometry['concepts']])
        adjacent=neighbors(matrices[name],lock['k'])
        details['models'][name]={'cosine_distances':distance_matrix(matrices[name]).tolist(),
                                'neighbors':[[geometry['concepts'][i] for i in row] for row in adjacent]}
    for pair in geometry['pairs']:
        scores=neighbor_scores(matrices[pair['left']],matrices[pair['right']],lock['k'])
        details['pairs'].append({'left':pair['left'],'right':pair['right'],
                                 'jaccard_by_concept':scores['jaccard'].tolist(),
                                 'overlap_by_concept':scores['overlap'].tolist()})
    write_json(output.with_name(output.stem+'_geometry_details.json'),details)
    with output.with_name(output.stem+'_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]))
        writer.writeheader();writer.writerows(predictions)
    result={'stage':'week8_locked_test','status':lock['status'],'lock_sha256':file_hash(lock_path),
            'dataset_hash':manifest['sample_hash'],'test_concepts':geometry['concepts'],
            'geometry_file':str(geometry_path),'category_axis_results':results,
            'limitations':['Released video labels; midpoint-frame validity not independently reviewed',
                           'Category axes are derived researcher labels, not physical state transitions',
                           'No identity-preservation claim applies to category-axis classification',
                           'Intervals and multiplicity adjustments belong to the next inference stage']}
    write_json(output,result)
    return result
