"""Run ROBUSTNESS_PROTOCOL.md using cached features only."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from alignment import neighbors, overlap
from experiment import load_features, load_manifest, digest, write_json

OUT = Path('results/robustness')

def fit_transform(train, mode, dimensions=128):
    if mode == 'baseline':
        return None, None, None
    mean = train.mean(0)
    if mode == 'center':
        return mean, None, None
    if mode != 'pca128':
        raise ValueError(mode)
    _, singular, vh = torch.linalg.svd(train-mean, full_matrices=False)
    basis = vh[:dimensions].T.contiguous()
    retained = float(singular[:dimensions].square().sum()/singular.square().sum())
    return mean, basis, retained

def apply_transform(x, fitted):
    mean, basis, _ = fitted
    if mean is None:
        return x
    x = x-mean
    if basis is not None:
        x = x@basis
    return F.normalize(x, dim=-1)

def measure(a, b, k, seed, permutations=1000):
    left, right = neighbors(a,k), neighbors(b,k)
    score = overlap(left,right)
    rng = np.random.default_rng(seed)
    null=[]
    for _ in range(permutations):
        p=rng.permutation(len(right))
        null.append(overlap(left,np.argsort(p)[right[p]]))
    mu=float(np.mean(null))
    return dict(score=score,null_mean=mu,null_q95=float(np.quantile(null,.95)),
                excess=score-mu,adjusted=(score-mu)/(1-mu),
                p=(1+sum(v>=score for v in null))/(1+permutations),null=null)

def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path('results/heldout_layers/results.json')
    previous=json.loads(source.read_text())
    run=Path('results/baseline')
    manifest=load_manifest(run)
    assert digest(manifest)==previous['manifest_hash']
    vectors={}
    for spec in manifest['config']['models']:
        name=spec['id']
        raw=load_features(run,spec,manifest)
        for s,split in enumerate(previous['splits']):
            layers=set()
            for r in previous['results']:
                if r['seed']==split['seed']:
                    if r['language']==name: layers.add(r['layers'][0])
                    if r['vision']==name: layers.add(r['layers'][1])
            threshold=previous['thresholds'][name][s]
            for layer in layers:
                train=F.normalize(raw[split['selection'],layer].clamp(-threshold,threshold),dim=-1)
                evaluation=F.normalize(raw[split['evaluation'],layer].clamp(-threshold,threshold),dim=-1)
                for mode in ['baseline','center','pca128']:
                    fitted=fit_transform(train,mode)
                    vectors[name,split['seed'],layer,mode]=(apply_transform(evaluation,fitted),fitted[2])
        del raw
        print('Prepared '+name,flush=True)
    results=[]
    for old in previous['results']:
        seed=old['seed']
        def get(mode):
            return [vectors[old[name],seed,old['layers'][i],mode] for i,name in enumerate(['language','vision'])]
        base=get('baseline')
        cached={}
        for n in [128,256,512]:
            for regime in ['fixed_k','fixed_fraction']:
                k=10 if regime=='fixed_k' else round(10*(n-1)/511)
                if (n,k) not in cached:
                    cached[n,k]=measure(base[0][0][:n],base[1][0][:n],k,seed+10000)
                results.append(dict(study='sample_size',mode=regime,n=n,k=k,fraction=k/(n-1),
                    seed=seed,language=old['language'],vision=old['vision'],layers=old['layers'],**cached[n,k]))
        assert np.isclose(cached[512,10]['score'],old['evaluation_score'],atol=1e-12)
        np.testing.assert_allclose(cached[512,10]['null'],old['null'],atol=1e-12)
        for mode in ['baseline','center','pca128']:
            data=get(mode)
            measured=cached[512,10] if mode=='baseline' else measure(data[0][0],data[1][0],10,seed+10000)
            results.append(dict(study='preprocessing',mode=mode,n=512,k=10,fraction=10/511,
                seed=seed,language=old['language'],vision=old['vision'],layers=old['layers'],
                retained_language=data[0][1],retained_vision=data[1][1],**measured))
        print('Scored '+str(seed)+' '+old['language']+' / '+old['vision'],flush=True)
    paths=['robustness.py','ROBUSTNESS_PROTOCOL.md','alignment.py',str(source)]
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    write_json(OUT/'results.json',dict(source_hashes=hashes,manifest_hash=digest(manifest),
        splits=previous['splits'],permutations=1000,results=results))
    fields=['study','mode','seed','language','vision','layers','n','k','fraction','score','null_mean','null_q95','excess','adjusted','p','retained_language','retained_vision']
    with (OUT/'scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    summarize(results)

def summarize(results):
    lines=['# Sample-size and preprocessing sensitivity','',
        'Three overlapping splits; six model pairs. Layers and clipping fixed from selection halves. Each condition has 1,000 matched permutations. Ranges are across dependent comparisons, not confidence intervals.','',
        '| Study | Condition | N | k | Raw range | Null mean range | Excess range | Strict size increases / 6 |',
        '|---|---|---:|---:|---|---|---|---:|']
    groups=[]
    for r in results:
        key=(r['study'],r['mode'],r['n'],r['k'])
        if key not in groups: groups.append(key)
    for study,mode,n,k in groups:
        rr=[r for r in results if (r['study'],r['mode'],r['n'],r['k'])==(study,mode,n,k)]
        monotonic=0
        for seed in sorted(set(r['seed'] for r in rr)):
            for size in ['small','base']:
                ordered=[next(r for r in rr if r['seed']==seed and size in r['vision'] and r['language'].endswith(lang)) for lang in ['560m','1b1','1b7']]
                monotonic+=all(a['score']<b['score'] for a,b in zip(ordered,ordered[1:]))
        ranges=['%.4f–%.4f'%(min(r[field] for r in rr),max(r[field] for r in rr)) for field in ['score','null_mean','excess']]
        lines.append(f'| {study} | {mode} | {n} | {k} | '+ ' | '.join(ranges)+f' | {monotonic} |')
    lines+=['','## Matched preprocessing differences','']
    for mode in ['center','pca128']:
        changes=[]
        for r in results:
            if r['study']=='preprocessing' and r['mode']==mode:
                base=next(b for b in results if b['study']=='preprocessing' and b['mode']=='baseline' and all(b[x]==r[x] for x in ['seed','language','vision']))
                changes.append(r['score']-base['score'])
        lines.append(f'- {mode}: score change mean {np.mean(changes):.6f}; range {min(changes):.6f} to {max(changes):.6f}; increases in {sum(x>0 for x in changes)}/18 comparisons.')
    lines+=['',f"All stored condition scores exceed their corresponding null 95th percentile: {all(r['score']>r['null_q95'] for r in results)}.",'',
        'N=512 sample-size regimes and preprocessing baseline duplicate the same measurements. Do not count them as new evidence. Fixed-fraction k is rounded (2,5,10). No claims about N>512, independent replications, or statistically established differences between sizes. Preprocessing is a geometry intervention, not a validated correction. PCA is fitted independently per modality, without whitening, at baseline-selected layers.', '',
        'All 18 baseline scores and all their null draws match the previous held-out run. Protocol: ROBUSTNESS_PROTOCOL.md. Reproduce: `.venv/bin/python robustness.py`. Full draws/provenance: results.json; detailed rows: scores.csv.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,mode in zip(axes,['fixed_k','fixed_fraction']):
        for size,color in [('small','tab:blue'),('base','tab:orange')]:
            values=[]
            for n in [128,256,512]:
                rr=[r['excess'] for r in results if r['study']=='sample_size' and r['mode']==mode and r['n']==n and size in r['vision']]
                values.append(rr)
            ax.plot([128,256,512],np.mean(values,axis=1),marker='o',color=color,label='DINOv2 '+size)
            ax.fill_between([128,256,512],np.min(values,axis=1),np.max(values,axis=1),alpha=.15,color=color)
        ax.set_title('Fixed k=10' if mode=='fixed_k' else 'Approximately fixed k/(N−1)')
        ax.set_xlabel('Evaluation pool N'); ax.legend(fontsize=8)
    axes[0].set_ylabel('Overlap minus shuffled mean')
    fig.suptitle('Means and ranges across three BLOOMZ sizes × three splits (not CIs)')
    fig.tight_layout(); fig.savefig(OUT/'sample_size.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,size in zip(axes,['small','base']):
        for mode in ['baseline','center','pca128']:
            values=[[r['excess'] for r in results if r['study']=='preprocessing' and r['mode']==mode and size in r['vision'] and r['language'].endswith(lang)] for lang in ['560m','1b1','1b7']]
            ax.plot(['560M','1.1B','1.7B'],np.mean(values,axis=1),marker='o',label=mode)
        ax.set_title('DINOv2 '+size); ax.set_xlabel('BLOOMZ size'); ax.legend(fontsize=8)
    axes[0].set_ylabel('Overlap minus shuffled mean')
    fig.suptitle('Preprocessing sensitivity: means across three overlapping splits')
    fig.tight_layout(); fig.savefig(OUT/'preprocessing.png',dpi=160); plt.close(fig)

if __name__=='__main__': main()
