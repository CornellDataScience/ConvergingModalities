"""Exploratory, post-primary object-preservation follow-up. Main results unchanged."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import digest,write_json
from difference_transfer import fit_ridge
from semantic_analysis import fit_preprocessing,load_selected
from state_analysis import RUN,METRICS,translated_scores,clustered_bootstrap,macro


def identity_basis(features,noun_labels,retention=.8):
    centers=torch.stack([features[[i for i,n in enumerate(noun_labels) if n==noun]].mean(0) for noun in sorted(set(noun_labels))])
    centers=centers-centers.mean(0)
    _,s,vh=torch.linalg.svd(centers,full_matrices=False)
    rank=int(torch.searchsorted(s.square().cumsum(0)/s.square().sum(),torch.tensor(retention,dtype=s.dtype)).item())+1
    return vh[:rank].T


def project_direction(direction,basis,strength):
    residual=direction-strength*(direction@basis)@basis.T
    return F.normalize(residual,dim=0)*direction.norm()


def choose_preserving(endpoints,direction,scales,basis=None):
    baseline=translated_scores(endpoints,direction,0)['noun'].mean()
    best=None;trials=[]
    for strength in ([0] if basis is None else [0,.5,1]):
        adjusted=direction if basis is None else project_direction(direction,basis,strength)
        for scale in scales:
            values=translated_scores(endpoints,adjusted,scale);ap=float(values['ap'].mean());loss=float(values['noun'].mean()-baseline)
            eligible=loss>=-.05
            trials.append(dict(strength=strength,scale=scale,ap=ap,noun_change=loss,eligible=eligible))
            if eligible and (best is None or ap>best['ap']):best=dict(strength=strength,scale=scale,ap=ap,direction=adjusted)
    return best,trials


def run():
    torch.set_num_threads(4);out=RUN/'secondary_identity';out.mkdir(exist_ok=True)
    protocol=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='exploratory: designed after inspecting primary state-transfer results',
                  methods=['validation-constrained scale','validation-constrained training-noun-subspace projection'],
                  parameters=dict(validation_noun_tolerance=.05,identity_variance_retention=.8,projection_strengths=[0,.5,1]),
                  selection='maximize validation AP subject to noun loss <=.05; ascending strength then scale tie break',
                  limits='Training noun centroids are proxies for identity, not a proven disentangled object subspace. No new confirmatory significance claims.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primary_results_sha256=hashlib.sha256((RUN/'results.json').read_bytes()).hexdigest())
    write_json(out/'design.json',protocol)
    m=json.loads((RUN/'manifest.json').read_text());config=m['config'];records=m['records'];groups=m['noun_groups'];rels=config['relations'];n=config['images_per_composition']
    rx,qx,xmeta=load_selected(RUN,'language');ry,qy,_=load_selected(RUN,'vision');prompts={t:i for i,t in enumerate(xmeta['items'])};xrows=[prompts[r['text']] for r in records]
    lookup={(r['state'],r['noun'],r['instance']):i for i,r in enumerate(records)}
    cells=[];choices=[]
    for fold in range(config['folds']):
        train_groups=set(range(config['folds']))-{fold,(fold+1)%config['folds']}
        fit=[i for i,r in enumerate(records) if groups[r['noun']] in train_groups and r['instance']<n//2]
        tx,_=fit_preprocessing(rx,qx,[xrows[i] for i in fit],config['dimensions']);ty,_=fit_preprocessing(ry,qy,fit,config['dimensions'])
        x,y=tx(rx),ty(ry);w=fit_ridge(x[[xrows[i] for i in fit]],y[fit],config['ridge_alpha'])
        basis=identity_basis(y[fit],[records[i]['noun'] for i in fit])
        def text(state,noun):return torch.stack([x[prompts[t.format(state=state,noun=noun)]] for t in config['templates']]).mean(0)
        def endpoints(r,nouns):return torch.stack([torch.stack([torch.stack([y[lookup[state,noun,i]] for state in [r['source'],r['target']]]) for i in range(n)]) for noun in nouns])
        for ri,r in enumerate(rels):
            train=[noun for noun in m['support'][r['name']] if groups[noun] in train_groups]
            val_nouns=[noun for noun in m['support'][r['name']] if groups[noun]==(fold+1)%config['folds']]
            test_nouns=[noun for noun in m['support'][r['name']] if groups[noun]==fold]
            d=torch.stack([text(r['target'],noun)-text(r['source'],noun) for noun in train]).mean(0)@w
            val,ev=endpoints(r,val_nouns),endpoints(r,test_nouns)
            values={}
            for name,b in [('constrained_scale',None),('identity_projection',basis)]:
                chosen,trials=choose_preserving(val,d,config['scales'],b)
                values[name]=translated_scores(ev,chosen['direction'],chosen['scale'])
                choices.append(dict(fold=fold,relation=r['name'],method=name,scale=chosen['scale'],strength=chosen['strength'],basis_rank=basis.shape[1],validation_trials=trials))
            for ni,noun in enumerate(test_nouns):cells.append(dict(relation=ri,noun=noun,values={name:{metric:np.asarray(value[metric][ni],dtype=float).tolist() for metric in METRICS} for name,value in values.items()}))
    primary=json.loads((RUN/'per_query.json').read_text())
    assert [(c['relation'],c['noun']) for c in primary]==[(c['relation'],c['noun']) for c in cells]
    rel_ids=np.array([c['relation'] for c in cells]);nouns=sorted({c['noun'] for c in cells});noun_ids=np.array([nouns.index(c['noun']) for c in cells])
    names=['constrained_scale','identity_projection'];score={method:{metric:np.array([c['values'][method][metric] for c in cells]) for metric in METRICS} for method in names}
    base={method:{metric:np.array([c['values'][method][metric] for c in primary]) for metric in METRICS} for method in ['no_shift','text']}
    effects=[];labels=[]
    for name in names:
        for ref in ['no_shift','text']:effects.append(score[name]['ap']-base[ref]['ap']);labels.append(name+'_ap_minus_'+ref)
        effects.append(score[name]['noun']-base['no_shift']['noun']);labels.append(name+'_noun_change')
    effects=np.stack(effects);boot,cat_boot,_,weights,assets=clustered_bootstrap(effects,rel_ids,noun_ids,config['seed']+5000,config['bootstrap_draws'],len(rels))
    result=dict(status=protocol['status'],choices=choices,conditions={name:{metric:float(macro(a,rel_ids,len(rels))) for metric,a in value.items()} for name,value in score.items()},
                exploratory_intervals={name:dict(effect=float(macro(effects[i],rel_ids,len(rels))),ci95=np.quantile(boot[i],[.025,.975]).tolist(),one_sided_lower95=float(np.quantile(boot[i],.05))) for i,name in enumerate(labels)},
                primary_results_sha256=protocol['primary_results_sha256'])
    write_json(out/'results.json',result);write_json(out/'per_query.json',cells);np.savez_compressed(out/'bootstrap.npz',bootstrap=boot,category_only=cat_boot,noun_weights=weights,query_draws=assets)
    print(json.dumps({'conditions':result['conditions'],'exploratory_intervals':result['exploratory_intervals']},indent=2))


if __name__=='__main__':run()
