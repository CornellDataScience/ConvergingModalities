"""Conditional paired query bootstrap; see PAIRED_PROTOCOL.md."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import bootstrap
from experiment import load_manifest, load_features, digest, write_json
from robustness import fit_transform, apply_transform

OUT=Path('results/paired_uncertainty')
CONTRASTS=['center_minus_baseline','pca_minus_baseline','pca_minus_center',
           'large_minus_small','pool_fixed_k_excess','pool_fixed_fraction_excess']

def reference_overlap(a,b,k,reference_size=256):
    """Vectors ordered: 256 reference rows followed by query rows."""
    na=torch.argsort(a[256:]@a[:reference_size].T,dim=1,descending=True)[:,:k].numpy()
    nb=torch.argsort(b[256:]@b[:reference_size].T,dim=1,descending=True)[:,:k].numpy()
    return (na[:,:,None]==nb[:,None,:]).any(axis=2).mean(axis=1)

def intervals(d,seed,resamples=20000,family=18):
    result=bootstrap((np.asarray(d),),np.mean,vectorized=True,n_resamples=resamples,
                     confidence_level=.95,method='BCa',rng=np.random.default_rng(seed),batch=500)
    adjusted=bootstrap((np.asarray(d),),np.mean,vectorized=True,n_resamples=0,
                       confidence_level=1-.05/family,method='BCa',bootstrap_result=result)
    return dict(effect=float(np.mean(d)),ci95=[float(x) for x in result.confidence_interval],
                ci_adjusted=[float(x) for x in adjusted.confidence_interval],
                bootstrap_se=float(result.standard_error),draws=result.bootstrap_distribution.tolist())

def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path('results/heldout_layers/results.json').read_text())
    manifest=load_manifest(Path('results/baseline'))
    assert digest(manifest)==previous['manifest_hash']
    vectors={}
    for spec in manifest['config']['models']:
        name=spec['id']; raw=load_features(Path('results/baseline'),spec,manifest)
        for s,split in enumerate(previous['splits']):
            relevant=[r for r in previous['results'] if r['seed']==split['seed'] and name in [r['language'],r['vision']]]
            layers={r['layers'][0 if r['language']==name else 1] for r in relevant}
            threshold=previous['thresholds'][name][s]
            for layer in layers:
                train=F.normalize(raw[split['selection'],layer].clamp(-threshold,threshold),dim=-1)
                ev=F.normalize(raw[split['evaluation'],layer].clamp(-threshold,threshold),dim=-1)
                for mode in ['baseline','center','pca128']:
                    vectors[name,split['seed'],layer,mode]=apply_transform(ev,fit_transform(train,mode))
        del raw
        print('Prepared '+name,flush=True)
    results=[]; scores=[]; split_ids=[]
    for split in previous['splits']:
        seed=split['seed']; pairs={}
        split_ids.append(dict(seed=seed,selection=split['selection'],reference=split['evaluation'][:256],query=split['evaluation'][256:]))
        for old in [r for r in previous['results'] if r['seed']==seed]:
            pair=(old['language'],old['vision']); values={}
            for mode in ['baseline','center','pca128']:
                a,b=[vectors[old[name],seed,old['layers'][i],mode] for i,name in enumerate(['language','vision'])]
                values[mode]=reference_overlap(a,b,10)
                if mode=='baseline':
                    values['pool128_k10']=reference_overlap(a,b,10,128)
                    values['pool128_k5']=reference_overlap(a,b,5,128)
            pairs[pair]=values
            scores.append(dict(seed=seed,language=pair[0],vision=pair[1],layers=old['layers'],per_query={k:v.tolist() for k,v in values.items()}))
        average=lambda key: np.mean([p[key] for p in pairs.values()],axis=0)
        large=[]; small=[]
        for vision in sorted({p[1] for p in pairs}):
            large.append(pairs['bigscience/bloomz-1b7',vision]['baseline'])
            small.append(pairs['bigscience/bloomz-560m',vision]['baseline'])
        diffs=[average('center')-average('baseline'),average('pca128')-average('baseline'),
               average('pca128')-average('center'),np.mean(large,axis=0)-np.mean(small,axis=0),
               average('baseline')-average('pool128_k10')-(10/256-10/128),
               average('baseline')-average('pool128_k5')]
        for name,d in zip(CONTRASTS,diffs):
            r=intervals(d,seed+20000)
            r.update(seed=seed,contrast=name,per_query_difference=d.tolist())
            results.append(r)
            print(seed,name,round(r['effect'],6),r['ci_adjusted'],flush=True)
    files=['paired_uncertainty.py','PAIRED_PROTOCOL.md','robustness.py','results/heldout_layers/results.json']
    write_json(OUT/'results.json',dict(manifest_hash=digest(manifest),source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
               splits=split_ids,bootstrap_resamples=20000,family_size=18,results=results,query_scores=scores))
    lines=['# Conditional paired difference uncertainty','',
        '256 query pairs, separate fixed bank of 256 reference pairs; selection/fitting uses another 512 pairs. Effects average across a fixed roster. BCa bootstrap with 20,000 paired query resamples. Family-adjusted intervals use confidence 1−0.05/18.','',
        '| Split | Contrast | Difference | 95% CI | Family-adjusted CI | Excludes zero after adjustment? |',
        '|---|---|---:|---|---|---|']
    for r in results:
        low,high=r['ci_adjusted']
        lines.append(f"| {r['seed']} | {r['contrast']} | {r['effect']:.5f} | [{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}] | [{low:.5f}, {high:.5f}] | {low>0 or high<0} |")
    lines+=['','## Interpretation limits','',
        'These intervals concern differences for the fixed reference banks and fixed fitted representations. They are not confidence intervals for the previous shared-pool metric, and exclude uncertainty from reference selection and model fitting. Splits overlap and the dataset was already inspected. Exchangeable-query assumptions and approximate bootstrap coverage are required. No p-values are inferred from bootstrap sign proportions. Pool contrasts remove the analytical chance shift; raw effects are recoverable from saved query scores.','',
        'Reproduce: `.venv/bin/python paired_uncertainty.py`. Exact rows, scores, draws and source hashes: results.json. Method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7))
    for s,seed in enumerate([x['seed'] for x in previous['splits']]):
        rr=[r for r in results if r['seed']==seed]
        y=np.arange(6)+(s-1)*.22
        means=np.array([r['effect'] for r in rr]); bounds=np.array([r['ci_adjusted'] for r in rr])
        ax.errorbar(means,y,xerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt='o',capsize=3,label=str(seed))
    ax.axvline(0,color='black',linewidth=1)
    ax.set_yticks(np.arange(6),CONTRASTS)
    ax.set_xlabel('Paired overlap difference (pool effects subtract chance shift)')
    ax.set_title('Conditional differences: Bonferroni-adjusted BCa intervals (18 contrasts)')
    ax.legend(title='Split');fig.tight_layout();fig.savefig(OUT/'paired.png',dpi=160);plt.close(fig)

if __name__=='__main__': main()
