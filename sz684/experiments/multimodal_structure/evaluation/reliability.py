"""Development-only repeatability, estimated without touching test scores."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from shared.io import write_json
from ..training.build_fits import load_study
from .rsa import rsa
from .cka import linear_cka


def split_half(features, ids, records, split, seed=20261023):
    """Compare concept RDMs from two disjoint sets of original sources.

    This is a descriptive split, not a corrected noise ceiling or confidence
    interval. Text halves measure template stability, not independent situations.
    """
    index={row['sample_id']:row for row in records}
    grouped=defaultdict(lambda:defaultdict(list))
    for value,sample in zip(features,ids):
        row=index[sample]
        if row['split']==split: grouped[row['concept_id']][row['source_id']].append(value)
    first,second,concepts=[],[],[]
    rng=np.random.default_rng(seed)
    for concept,sources in sorted(grouped.items()):
        order=rng.permutation(sorted(sources))
        if len(order)<2: continue
        middle=len(order)//2
        first.append(np.mean([np.mean(sources[source],axis=0) for source in order[:middle]],axis=0))
        second.append(np.mean([np.mean(sources[source],axis=0) for source in order[middle:]],axis=0))
        concepts.append(concept)
    if len(concepts)<3: raise ValueError('At least three concepts need independent source halves')
    return {'concepts':len(concepts),'rdm_spearman':rsa(np.stack(first),np.stack(second)),
            'linear_cka':linear_cka(np.stack(first),np.stack(second))}


def development_reliability(manifest_path, cache_paths, output):
    """Save only train/validation repeatability; test is reserved for the lock."""
    manifest,data,_=load_study(manifest_path,cache_paths)
    scores={}
    for name,(features,ids) in data.items():
        scores[name]={split:split_half(features,ids,manifest['records'],split) for split in ['train','validation']}
    result={'dataset_hash':manifest['sample_hash'],'models':scores,
            'status':'Descriptive source split-half stability; no test scores or intervals'}
    write_json(Path(output),result)
    return result
