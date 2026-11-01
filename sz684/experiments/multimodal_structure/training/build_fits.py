"""Week 7: fit on training sources, select settings on validation, never score test."""
import json
from pathlib import Path
from itertools import permutations
import numpy as np
from shared.io import write_json, digest
from ..provenance import file_hash,software_hashes
from ..data.prepare import load_manifest
from ..data.axes import AXES, axis_label
from ..extraction.cache import load_features
from ..evaluation.evaluate import concept_centroids
from ..evaluation.directions import evaluate_transfer
from ..evaluation.fit_directions import json_arrays
from .preprocessing import fit_pca, transform
from .alignment import fit_procrustes
from .probes import fit_probe, predict, balanced_accuracy


def source_partition(records):
    """Within each training concept, keep anchor and probe sources disjoint."""
    grouped={}
    for row in records:
        if row['split']=='train': grouped.setdefault(row['concept_id'],set()).add(row['source_id'])
    anchors, probes=set(),set()
    for concept,sources in grouped.items():
        order=sorted(sources)
        if len(order)<2: raise ValueError('Need separate anchor/probe sources per training concept')
        midpoint=len(order)//2
        anchors.update(order[:midpoint]);probes.update(order[midpoint:])
    if anchors & probes: raise ValueError('One source occurs in multiple training concepts/partitions')
    return anchors,probes


def select_axis(features, ids, index, axis, split, sources=None):
    """Select rows and binary researcher-defined category labels explicitly."""
    selected,labels=[],[]
    for i,sample in enumerate(ids):
        row=index[sample]
        if row['split']!=split: continue
        if sources is not None and row['source_id'] not in sources: continue
        label=axis_label(axis,row['family_id'])
        if label is not None:
            selected.append(i);labels.append(label)
    if set(labels)!={0,1}: raise ValueError(f'{axis}: both labels are required in {split}')
    return features[selected],np.array(labels)


def load_study(manifest_path, cache_paths):
    """Check complete row coverage, model modalities and shared dataset identity."""
    manifest=load_manifest(manifest_path)
    data,identities={},{}
    for name,path in cache_paths.items():
        features,ids,identity=load_features(path)
        if identity['dataset_hash']!=manifest['sample_hash']: raise ValueError('Cache dataset differs')
        modality=identity['model']['modality']
        expected={row['sample_id'] for row in manifest['records'] if row['modality']==modality}
        if set(ids)!=expected: raise ValueError('Cache must cover every row of its modality')
        data[name]=(features,ids);identities[name]=identity
    return manifest,data,identities


def build_fits(manifest_path, cache_paths, output, dimensions=(8,16,32), seed=20261023):
    """Save all selected parameters before any test scores are computed.

    Dimension is selected per map direction by mean validation transfer accuracy
    across the two declared category axes. Probe alpha is selected per model,
    dimension and axis. Ties prefer smaller dimensions/regularization.
    """
    output=Path(output)
    if output.exists(): raise ValueError('Fitting output exists; use a new run')
    manifest,data,identities=load_study(manifest_path,cache_paths)
    index={row['sample_id']:row for row in manifest['records']}
    transforms,probes,anchors,partitions={},{},{},{}
    for name,(features,ids) in data.items():
        rows=[index[sample] for sample in ids]
        anchor_sources,probe_sources=source_partition(rows)
        partitions[name]={'anchor_sources':sorted(anchor_sources),'probe_sources':sorted(probe_sources)}
        anchor_rows=[row for row in rows if row['split']=='train' and row['source_id'] in anchor_sources]
        selected=[i for i,sample in enumerate(ids) if index[sample]['source_id'] in anchor_sources and index[sample]['split']=='train']
        centroids=concept_centroids(features[selected],[ids[i] for i in selected],anchor_rows,'train')
        concepts=sorted(centroids)
        anchor_matrix=np.stack([centroids[concept] for concept in concepts])
        transforms[name],probes[name],anchors[name]={},{},{}
        for width in sorted(dimensions):
            try: fitted=fit_pca(anchor_matrix,width)
            except ValueError: continue
            key=str(width)
            transforms[name][key]=fitted
            anchors[name][key]={'concepts':concepts,'features':transform(anchor_matrix,fitted)}
            projected=transform(features,fitted)
            probes[name][key]={}
            for axis in AXES:
                training,labels=select_axis(projected,ids,index,axis,'train',probe_sources)
                validation,val_labels=select_axis(projected,ids,index,axis,'validation')
                candidates=[fit_probe(training,labels,alpha) for alpha in [.1,1.,10.]]
                scores=[balanced_accuracy(val_labels,predict(validation,probe)) for probe in candidates]
                best=int(np.argmax(scores))
                probes[name][key][axis]=dict(candidates[best],validation_accuracy=scores[best])
    pairs=[]
    for source,target in permutations(data,2):
        candidates=[]
        for key in sorted(set(transforms[source]) & set(transforms[target]),key=int):
            left,right=anchors[source][key],anchors[target][key]
            if left['concepts']!=right['concepts']: raise ValueError('Training anchor concept mismatch')
            alignment=fit_procrustes(left['features'],right['features'])
            shuffled=fit_procrustes(left['features'],right['features'][np.random.default_rng(seed).permutation(len(right['features']))])
            target_features,target_ids=data[target]
            projected=transform(target_features,transforms[target][key])
            scores={}
            for axis in AXES:
                validation,labels=select_axis(projected,target_ids,index,axis,'validation')
                scores[axis]=evaluate_transfer(probes[source][key][axis],probes[target][key][axis],alignment,validation,labels)
            objective=float(np.mean([score['transfer_balanced_accuracy'] for score in scores.values()]))
            candidates.append({'dimensions':int(key),'alignment':alignment,'shuffled_alignment':shuffled,
                               'validation':scores,'objective':objective})
        if not candidates: raise ValueError('No common effective rank for '+source+' / '+target)
        chosen=candidates[int(np.argmax([row['objective'] for row in candidates]))]
        pairs.append(dict(source=source,target=target,**chosen,
                          dimension_curve=[{'dimensions':row['dimensions'],'objective':row['objective']} for row in candidates]))
    result={'schema_version':2,'stage':'week7_train_validation_only','dataset_hash':manifest['sample_hash'],
            'models':identities,'feature_sha256':{name:file_hash(path) for name,path in cache_paths.items()},
            'software_hashes':software_hashes(),'axes':AXES,'dimension_grid':list(dimensions),'seed':seed,
            'partitions':partitions,'transforms':json_arrays(transforms),'probes':json_arrays(probes),
            'pairs':json_arrays(pairs),'test_scores_computed':False}
    write_json(output,result)
    return result
