"""Category-held-out RGB intervention transfer with clustered paired inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import digest, write_json
from difference_transfer import fit_ridge


def fit_preprocessing(raw, q95, rows, dimensions):
    threshold=q95[rows].mean()
    normalized=F.normalize(raw.clamp(-threshold,threshold),dim=-1)
    mean=normalized[rows].mean(0)
    _,singular,vh=torch.linalg.svd(normalized[rows]-mean,full_matrices=False)
    if len(singular)<dimensions or singular[dimensions-1] <= singular[0]*1e-8:
        raise ValueError('Insufficient effective rank for declared PCA dimension')
    basis=vh[:dimensions].T.contiguous()
    def transform(x):
        return F.normalize((F.normalize(x.clamp(-threshold,threshold),dim=-1)-mean)@basis,dim=-1)
    return transform,dict(threshold=float(threshold),retained_variance=float(singular[:dimensions].square().sum()/singular.square().sum()),
                          smallest_retained_relative_singular=float(singular[dimensions-1]/singular[0]))


def load_selected(run, modality):
    folder=run/('features_'+modality);meta=json.loads((folder/'metadata.json').read_text())
    if not meta['complete']:raise ValueError('Incomplete feature cache')
    arrays=[];quantiles=[];rows=[]
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as a:
            arrays.append(a['features']);quantiles.append(a['q95']);rows.extend(a['rows'].tolist())
    if rows!=list(range(len(rows))):raise ValueError('Missing or repeated feature rows')
    return torch.tensor(np.concatenate(arrays),dtype=torch.float64),torch.tensor(np.concatenate(quantiles),dtype=torch.float64),meta


def margins(source,target,direction,scale):
    queries=F.normalize(source+scale*direction,dim=-1)
    return ((queries*target).sum(-1)-(queries*source).sum(-1)).numpy()


def choose_scale(source,target,direction,scales):
    values=[float(margins(source,target,direction,s).mean()) for s in scales]
    best=int(np.argmax(values))
    return scales[best],values


def bank_allowed(categories, instances):
    # Row order is category, instance; bank order is category, instance, red/blue.
    q=np.tile(np.arange(instances),categories)
    bank_instance=np.repeat(np.tile(np.arange(instances),categories),2)
    bank_color=np.tile([0,1],categories*instances)
    return ~(((bank_color[None,:]==0)&(bank_instance[None,:]==q[:,None]))|
             ((bank_color[None,:]==1)&(bank_instance[None,:]==((q+1)%instances)[:,None])))


def retrieval(queries, endpoints):
    c,n,_,d=endpoints.shape
    flat=queries.reshape(c*n,d);bank=endpoints.reshape(c*n*2,d)
    score=(flat@bank.T).numpy();allowed=bank_allowed(c,n)
    query_cat=np.repeat(np.arange(c),n);bank_cat=np.repeat(np.arange(c),n*2)
    truth=np.arange(c*n)*2+1
    output={}
    for label,mask in [('global',allowed),('within_category',allowed&(query_cat[:,None]==bank_cat[None,:]))]:
        masked=np.where(mask,score,-np.inf);winner=masked.argmax(1)
        rank=1+(masked>score[np.arange(c*n),truth,None]).sum(1)
        # np indexing above must yield one true score per query, not a broadcast row.
        output[label]=dict(color=(winner%2==1).reshape(c,n),category=(winner//(n*2)==query_cat).reshape(c,n),
                           joint=((winner%2==1)&(winner//(n*2)==query_cat)).reshape(c,n),
                           paired_top1=(winner==truth).reshape(c,n),paired_rank=rank.reshape(c,n))
    return output


def bootstrap_paired(effects,seed,draws):
    """Shared hierarchical draws for [contrast,category,drawing] paired effects."""
    rng=np.random.default_rng(seed);k,c,n=effects.shape
    category=rng.integers(c,size=(draws,c),dtype=np.uint8)
    instance=rng.integers(n,size=(draws,c,n),dtype=np.uint8)
    boot=np.empty((k,draws));category_only=np.empty((k,draws))
    for start in range(0,draws,500):
        cats=category[start:start+500];assets=instance[start:start+500]
        for i in range(k):
            boot[i,start:start+len(cats)]=effects[i][cats[:,:,None],assets].mean((1,2))
            category_only[i,start:start+len(cats)]=effects[i].mean(1)[cats].mean(1)
    return boot,category_only,category,instance


def summarize_retrieval(values):
    return {bank:{metric:float(np.mean(array)) for metric,array in stats.items()} for bank,stats in values.items()}


def analyze(stage,reverse=False,pilot_rotation=0):
    torch.set_num_threads(4);run=Path('results/semantic_'+stage)
    manifest=json.loads((run/'manifest.json').read_text());config=manifest['config'];spec=config[stage]
    if pilot_rotation:
        if stage!='pilot': raise ValueError('Rotations are retired pilot planning only')
        cats=spec['train']+spec['validation']+spec['test'];cats=cats[pilot_rotation:]+cats[:pilot_rotation]
        spec=dict(spec,train=cats[:2],validation=cats[2:3],test=cats[3:])
    records=manifest['records'];train=[i for i,r in enumerate(records) if r['category'] in spec['train'] and r['instance']<spec['instances']//2]
    rawx,qx,xmeta=load_selected(run,'language');rawy,qy,ymeta=load_selected(run,'vision')
    assert len(rawy)==len(records)
    for meta in [xmeta,ymeta]:assert meta['identity']['manifest_hash']==digest(manifest)
    prompt_index={t:i for i,t in enumerate(xmeta['items'])}
    xrows=[prompt_index[r['text']] for r in records]
    tx,xf=fit_preprocessing(rawx,qx,[xrows[i] for i in train],spec['dimensions'])
    ty,yf=fit_preprocessing(rawy,qy,train,spec['dimensions'])
    allx=tx(rawx);ally=ty(rawy);x=allx[[xrows[i] for i in train]];y=ally[train]
    w=fit_ridge(x,y,alpha=config['ridge_alpha']);xc=x-x.mean(0);yc=y-y.mean(0)
    u,_,vh=torch.linalg.svd(xc.T@yc,full_matrices=False);orthogonal=u@vh
    colors=['red','blue'] if not reverse else ['blue','red']
    def text_pair(cat):
        return torch.stack([torch.stack([allx[prompt_index[t.format(category=cat,color=color)]] for t in config['templates']]).mean(0) for color in colors])
    per_category=torch.stack([text_pair(c)[1]-text_pair(c)[0] for c in spec['train']])
    text_direction=per_category.mean(0)@w;procrustes_direction=per_category.mean(0)@orthogonal
    def endpoints(split):
        cats=spec[split];groups=[]
        for cat in cats:
            groups.append(torch.stack([torch.stack([ally[next(i for i,r in enumerate(records) if r['category']==cat and r['instance']==j and r['color']==color)] for color in colors]) for j in range(spec['instances'])]))
        return torch.stack(groups)
    train_y=endpoints('train');validation=endpoints('validation');test=endpoints('test')
    image_direction=(train_y[:,spec['instances']//2:,1]-train_y[:,spec['instances']//2:,0]).mean((0,1))
    rng=torch.Generator().manual_seed(config['seed']+int(reverse))
    random=F.normalize(torch.randn(config['random_controls'],spec['dimensions'],generator=rng,dtype=torch.float64),dim=-1)*text_direction.norm()
    signs=torch.randint(2,(config['random_controls'],len(per_category)),generator=rng,dtype=torch.int64)*2-1
    signed=(signs.to(torch.float64)@per_category/len(per_category))@w
    signed=F.normalize(signed,dim=-1)*text_direction.norm()
    source,target=test[:,:,0],test[:,:,1];vs,vt=validation[:,:,0],validation[:,:,1]
    text_scale,text_validation=choose_scale(vs,vt,text_direction,config['scales'])
    image_scale,image_validation=choose_scale(vs,vt,image_direction,config['scales'])
    proc_scale,proc_validation=choose_scale(vs,vt,procrustes_direction,config['scales'])
    directions={'no_shift':(text_direction,0),'text':(text_direction,text_scale),'reverse':(-text_direction,text_scale),
                'image_reference':(image_direction,image_scale),'procrustes':(procrustes_direction,proc_scale)}
    scores={name:margins(source,target,d,s) for name,(d,s) in directions.items()}
    random_scores=np.stack([margins(source,target,d,text_scale) for d in random])
    sign_scores=np.stack([margins(source,target,d,text_scale) for d in signed])
    scores['mean_random']=random_scores.mean(0);scores['mean_sign_randomized']=sign_scores.mean(0)
    retrieved={name:retrieval(F.normalize(source+s*d,dim=-1),test) for name,(d,s) in directions.items()}
    direct=torch.stack([(text_pair(cat)[1]-x.mean(0))@w+y.mean(0) for cat in spec['test']])
    direct=F.normalize(direct,dim=-1)[:,None,:].expand_as(source)
    retrieved['direct_target_text']=retrieval(direct,test)
    guard=retrieved['text']['global']['category'].astype(float)-retrieved['no_shift']['global']['category'].astype(float)
    effects=np.stack([scores['text']-scores[control] for control in ['no_shift','mean_random','reverse']]+[guard])
    boot,cat_boot,cat_draws,asset_draws=bootstrap_paired(effects,config['seed']+10+int(reverse),config['bootstrap_draws'])
    names=['text_minus_no_shift','text_minus_random','text_minus_reverse','category_preservation']
    inference={}
    for k,name in enumerate(names):
        cat_effect=effects[k].mean(1);alpha=.05/3
        inference[name]=dict(effect=float(effects[k].mean()),ci95=np.quantile(boot[k],[.025,.975]).tolist(),
                             ci_adjusted=np.quantile(boot[k],[alpha/2,1-alpha/2]).tolist() if k<3 else None,
                             one_sided_lower95=float(np.quantile(boot[k],.05)),category_effects=cat_effect.tolist(),
                             category_only_ci95=np.quantile(cat_boot[k],[.025,.975]).tolist(),
                             leave_one_category_out=[float(np.delete(cat_effect,i).mean()) for i in range(len(cat_effect))] if len(cat_effect)>1 else None)
    success=all(inference[name]['ci_adjusted'][0]>0 for name in names[:3]) and inference[names[3]]['one_sided_lower95']> -config['category_preservation_tolerance']
    alignment=F.cosine_similarity(text_direction[None,None,:],target-source,dim=-1).numpy()
    out=run/(('rotation_'+str(pilot_rotation)) if pilot_rotation else ('reverse' if reverse else 'forward'));out.mkdir(exist_ok=True)
    np.savez_compressed(out/'draws.npz',category_draws=cat_draws,asset_draws=asset_draws,bootstrap=boot,category_only_bootstrap=cat_boot)
    np.savez_compressed(out/'per_query.npz',**scores,random_scores=random_scores,sign_randomized_scores=sign_scores,category_preservation=guard,direction_cosine=alignment,
                        **{name+'_'+bank+'_'+metric:a for name,banks in retrieved.items() for bank,stats in banks.items() for metric,a in stats.items()})
    np.savez(out/'vectors.npz',source=source.numpy(),target=target.numpy(),text_direction=text_direction.numpy(),image_direction=image_direction.numpy(),ridge_map=w.numpy())
    results=dict(stage=stage,relation='→'.join(colors),confirmatory_scope='local prospective primary analysis' if stage=='main' and not reverse else 'pilot/descriptive secondary',
                 manifest_hash=digest(manifest),categories=spec['test'],instances_per_category=spec['instances'],dimensions=spec['dimensions'],
                 transforms={'language':xf,'vision':yf},scales={'text':text_scale,'image':image_scale,'procrustes':proc_scale},
                 validation_curves={'text':text_validation,'image':image_validation,'procrustes':proc_validation},
                 direction_norms={'text':float(text_direction.norm()),'image':float(image_direction.norm())},
                 margins={name:float(a.mean()) for name,a in scores.items()},inference=inference,
                 retrieval={name:summarize_retrieval(a) for name,a in retrieved.items()},
                 mean_direction_cosine=float(alignment.mean()),passes_predeclared_criteria=bool(success),
                 source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['QUICKDRAW_PROTOCOL.md','semantic_analysis.py','semantic_extract.py','quickdraw_data.py','configs/semantic_quickdraw.json']})
    write_json(out/'results.json',results)
    lines=[f'# Controlled sketch {results["relation"]} — {stage}', '',f'Categories: {len(spec["test"])}; drawings per category: {spec["instances"]}; PCA: {spec["dimensions"]}.', '',
           f'Validation scales: {results["scales"]}. Criteria pass: {success}. Pilot/secondary results are descriptive only.', '',
           '| Contrast | Paired effect | 95% interval | Adjusted interval (3 contrasts) |','|---|---:|---|---|']
    for name,stats in inference.items():lines.append(f'| {name} | {stats["effect"]:.5f} | {stats["ci95"]} | {stats["ci_adjusted"]} |')
    lines+=['','| Method | Margin | Color top1 | Category top1 | Joint top1 | Matched blue top1 |','|---|---:|---:|---:|---:|---:|']
    for name,value in results['retrieval'].items():
        a=value['global'];lines.append(f'| {name} | {results["margins"].get(name,float("nan")):.4f} | {a["color"]:.2%} | {a["category"]:.2%} | {a["joint"]:.2%} | {a["paired_top1"]:.2%} |')
    lines+=['',f'Mean cosine of text direction and actual paired image changes: {results["mean_direction_cosine"]:.4f}.', '',
            'Synthetic RGB recoloring of game-recognized sketches; no human relabeling. Unknown author clustering, prefix sampling, fixed category roster and approximate bootstrap coverage limit inference. Retrieval excludes each red query but contains its near-identical blue counterfactual, so no-shift performance can be near ceiling. Passing these criteria is limited evidence for this controlled intervention, not natural-image analogy.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:results[k] for k in ['stage','relation','scales','margins','inference','passes_predeclared_criteria']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','main'],required=True);p.add_argument('--reverse',action='store_true');p.add_argument('--pilot-rotation',type=int,choices=range(4),default=0);a=p.parse_args();analyze(a.stage,a.reverse,a.pilot_rotation)
