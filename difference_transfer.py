"""Supervised held-out pair-difference pilot; not a semantic relation benchmark."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import load_features,load_manifest,digest,write_json
from robustness import fit_transform,apply_transform

OUT=Path('results/difference_transfer')

def fit_ridge(x,y,alpha=1.):
    xc=x-x.mean(0);yc=y-y.mean(0)
    return torch.linalg.solve(xc.T@xc+alpha*torch.eye(x.shape[1],dtype=x.dtype),xc.T@yc)

def difference_scores(predicted,target):
    similarities=F.normalize(predicted,dim=-1)@F.normalize(target,dim=-1).T
    ranks=torch.argsort(similarities,dim=1,descending=True)
    truth=torch.arange(len(target))
    return dict(mean_cosine=float(similarities.diag().mean()),
                top1=float((ranks[:,0]==truth).float().mean()),
                top5=float((ranks[:,:5]==truth[:,None]).any(1).float().mean()))

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    source=Path('results/heldout_layers/results.json');old=json.loads(source.read_text())
    split=old['splits'][0];seed=split['seed']
    language='bigscience/bloomz-1b7';vision='timm/vit_base_patch14_dinov2.lvd142m'
    row=next(r for r in old['results'] if r['seed']==seed and r['language']==language and r['vision']==vision)
    manifest=load_manifest(Path('results/baseline'));assert digest(manifest)==old['manifest_hash']
    data=[]
    for i,name in enumerate([language,vision]):
        spec=next(m for m in manifest['config']['models'] if m['id']==name)
        raw=load_features(Path('results/baseline'),spec,manifest)
        threshold=old['thresholds'][name][0];layer=row['layers'][i]
        train=F.normalize(raw[split['selection'],layer].clamp(-threshold,threshold),dim=-1)
        ev=F.normalize(raw[split['evaluation'],layer].clamp(-threshold,threshold),dim=-1)
        transform=fit_transform(train,'pca128')
        data.append((apply_transform(train,transform),apply_transform(ev,transform)))
        del raw
    (x,xe),(y,ye)=data
    dx=xe[1::2]-xe[::2];dy=ye[1::2]-ye[::2]
    w=fit_ridge(x,y);predicted=dx@w
    observed=difference_scores(predicted,dy)
    generator=np.random.default_rng(20261007)
    train_null=[]
    # Algebraically equivalent to refitting ridge for each permuted target matrix.
    xc=x-x.mean(0);yc=y-y.mean(0)
    solver=torch.linalg.solve(xc.T@xc+torch.eye(x.shape[1]),xc.T)
    for _ in range(199):
        perm=generator.permutation(len(y))
        shuffled_w=solver@yc[perm]
        train_null.append(difference_scores(dx@shuffled_w,dy))
    evaluation_null=[]
    for _ in range(999):
        perm=generator.permutation(len(dy))
        evaluation_null.append(difference_scores(predicted,dy[perm]))
    stats={}
    for metric in ['mean_cosine','top1','top5']:
        primary_p=(1+sum(r[metric]>=observed[metric] for r in train_null))/200
        stats[metric]=dict(observed=observed[metric],training_null_mean=float(np.mean([r[metric] for r in train_null])),
            training_tail_fraction_diagnostic=primary_p,
            evaluation_null_mean=float(np.mean([r[metric] for r in evaluation_null])),
            evaluation_p=(1+sum(r[metric]>=observed[metric] for r in evaluation_null))/1000,
            evaluation_p_bonferroni=min(1,2*(1+sum(r[metric]>=observed[metric] for r in evaluation_null))/1000) if metric!='top5' else None)
    paths=['difference_transfer.py','TRANSFER_PROTOCOL.md',str(source),'robustness.py']
    write_json(OUT/'results.json',dict(manifest_hash=digest(manifest),source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        language=language,vision=vision,layers=row['layers'],training_rows=split['selection'],
        evaluation_endpoint_pairs=np.array(split['evaluation']).reshape(-1,2).tolist(),
        ridge_alpha=1.,pca_dimensions=128,stats=stats,training_null=train_null,evaluation_null=evaluation_null))
    np.savez(OUT/'directions.npz',predicted=predicted.numpy(),target=dy.numpy(),map=w.numpy())
    lines=['# Supervised held-out difference-vector pilot','',
        '**This tests arbitrary disjoint pair differences, not repeated semantic concept relations.**', '',
        'BLOOMZ 1.7B → DINOv2 base; PCA128; supervised ridge map fit on 512 selection pairs; 256 disjoint evaluation endpoint pairs.','',
        '| Metric | Observed | Shuffled training-map mean | Evaluation-correspondence p (adjusted for 2 metrics) |',
        '|---|---:|---:|---:|']
    for metric,r in stats.items():
        lines.append(f"| {metric} | {r['observed']:.5f} | {r['training_null_mean']:.5f} | {r['evaluation_p_bonferroni']} |")
    lines+=['','Top-5 is descriptive, not part of the primary significance family. Chance top-1 retrieval is 1/256 ≈ 0.003906. Full null results are in results.json. Training shuffle tail fractions are diagnostic only because layer selection used training correspondences.','',
        'Post-run inferential correction: training shuffle tail fractions are not treated as valid p-values, because layers were selected using the true training pairing. Evaluation correspondence shuffles provide the inferential reference. This correction changed reporting, not embeddings, maps, scores, or permutation draws. A successful supervised endpoint map can yield successful difference transfer without a reusable semantic relation. No named relation labels were available; the semantic benchmark is specified in TRANSFER_PROTOCOL.md and remains unrun.','',
        'Reproduce: `.venv/bin/python difference_transfer.py`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
