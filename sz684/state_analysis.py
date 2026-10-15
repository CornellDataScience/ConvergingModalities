"""Natural object-state retrieval: five held-out-noun folds and paired inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiment import digest,write_json
from difference_transfer import fit_ridge
from semantic_analysis import fit_preprocessing,load_selected,bank_allowed

RUN=Path('results/state_transfer')
METRICS=['ap','within_ap','state','noun','joint','within_auc']


def score_queries(queries,endpoints):
    c,n,_,d=endpoints.shape; q=queries.reshape(c*n,d)
    score=(F.normalize(q,dim=-1)@endpoints.reshape(c*n*2,d).T).numpy()
    allowed=bank_allowed(c,n)
    qcat=np.repeat(np.arange(c),n);bcat=np.repeat(np.arange(c),n*2);bstate=np.tile([0,1],c*n)
    relevant=allowed&(qcat[:,None]==bcat[None,:])&(bstate[None,:]==1)
    masked=np.where(allowed,score,-np.inf);winner=masked.argmax(1)
    def ap(values,valid):
        truth=relevant&valid;order=np.argsort(-np.where(valid,values,-np.inf),axis=1,kind='stable')
        ranked=np.take_along_axis(truth,order,axis=1)
        precision=ranked.cumsum(1)/np.arange(1,ranked.shape[1]+1)
        return (precision*ranked).sum(1)/truth.sum(1)
    same=qcat[:,None]==bcat[None,:]
    auc=[]
    for i in range(c*n):
        positive=score[i,relevant[i]];negative=score[i,allowed[i]&same[i]&(bstate==0)]
        auc.append(((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])).mean())
    return dict(ap=ap(score,allowed).reshape(c,n),within_ap=ap(score,allowed&same).reshape(c,n),
                state=(bstate[winner]==1).reshape(c,n),noun=(bcat[winner]==qcat).reshape(c,n),
                joint=((bstate[winner]==1)&(bcat[winner]==qcat)).reshape(c,n),within_auc=np.array(auc).reshape(c,n))


def translated_scores(endpoints,direction,scale):
    return score_queries(endpoints[:,:,0]+scale*direction,endpoints)


def select_scale(endpoints,direction,scales):
    values=[float(translated_scores(endpoints,direction,s)['ap'].mean()) for s in scales]
    return scales[int(np.argmax(values))],values


def macro(values,relation_ids,relation_count):
    return np.mean([values[relation_ids==r].mean() for r in range(relation_count)])


def clustered_bootstrap(effects,relation_ids,noun_ids,seed,draws,relation_count):
    """Shared noun multiplicities across relations; queries resampled within cells."""
    k,cells,n=effects.shape;unique_nouns=int(noun_ids.max())+1;rng=np.random.default_rng(seed)
    weights=np.empty((draws,unique_nouns),dtype=np.uint8)
    for b in range(draws):
        while True:
            count=np.bincount(rng.integers(unique_nouns,size=unique_nouns),minlength=unique_nouns)
            if all(count[noun_ids[relation_ids==r]].sum()>0 for r in range(relation_count)):break
        weights[b]=count
    asset=rng.integers(n,size=(draws,cells,n),dtype=np.uint8)
    result=np.empty((k,draws));cat_only=np.empty_like(result);by_relation=np.empty((k,draws,relation_count))
    means=effects.mean(2)
    for start in range(0,draws,500):
        a=asset[start:start+500];w=weights[start:start+500][:,noun_ids]
        for contrast in range(k):
            sampled=np.take_along_axis(effects[contrast][None,:,:],a,axis=2).mean(2)
            estimates=[];category_estimates=[]
            for r in range(relation_count):
                selected=relation_ids==r;denom=w[:,selected].sum(1)
                value=(sampled[:,selected]*w[:,selected]).sum(1)/denom
                estimates.append(value);category_estimates.append((means[contrast,selected]*w[:,selected]).sum(1)/denom)
                by_relation[contrast,start:start+len(a),r]=value
            result[contrast,start:start+len(a)]=np.mean(estimates,axis=0)
            cat_only[contrast,start:start+len(a)]=np.mean(category_estimates,axis=0)
    return result,cat_only,by_relation,weights,asset


def analyze():
    torch.set_num_threads(4);manifest=json.loads((RUN/'manifest.json').read_text());config=manifest['config']
    records=manifest['records'];n=config['images_per_composition'];rels=config['relations'];groups=manifest['noun_groups']
    rawx,qx,xmeta=load_selected(RUN,'language');rawy,qy,ymeta=load_selected(RUN,'vision')
    for meta in [xmeta,ymeta]:assert meta['identity']['manifest_hash']==digest(manifest)
    assert len(rawy)==len(records)
    prompt_index={t:i for i,t in enumerate(xmeta['items'])};xrows=[prompt_index[r['text']] for r in records]
    index={(r['state'],r['noun'],r['instance']):i for i,r in enumerate(records)}
    per_cell=[];fold_info=[];vectors=[];condition_names=['no_shift','text','reverse','mean_random','mean_wrong','image_reference','procrustes','direct_target_text','noun_specific_text']
    for fold in range(config['folds']):
        train_groups={f for f in range(config['folds']) if f not in [fold,(fold+1)%config['folds']]}
        training=[i for i,r in enumerate(records) if r['group'] in train_groups and r['instance']<n//2]
        tx,xf=fit_preprocessing(rawx,qx,[xrows[i] for i in training],config['dimensions'])
        ty,yf=fit_preprocessing(rawy,qy,training,config['dimensions'])
        allx,ally=tx(rawx),ty(rawy);x=allx[[xrows[i] for i in training]];y=ally[training]
        w=fit_ridge(x,y,config['ridge_alpha']);xm,ym=x.mean(0),y.mean(0)
        u,_,vh=torch.linalg.svd((x-xm).T@(y-ym),full_matrices=False);procrustes=u@vh
        def text(state,noun):
            return torch.stack([allx[prompt_index[t.format(state=state,noun=noun)]] for t in config['templates']]).mean(0)
        def endpoints(relation,nouns):
            return torch.stack([torch.stack([torch.stack([ally[index[state,noun,i]] for state in [relation['source'],relation['target']]]) for i in range(n)]) for noun in nouns])
        train_nouns={r['name']:[noun for noun in manifest['support'][r['name']] if groups[noun] in train_groups] for r in rels}
        text_offsets={r['name']:torch.stack([text(r['target'],noun)-text(r['source'],noun) for noun in train_nouns[r['name']]]).mean(0) for r in rels}
        mapped={name:d@w for name,d in text_offsets.items()}
        saved=dict(fold=fold,training_nouns=sorted(noun for noun,group in groups.items() if group in train_groups),
                   test_nouns=sorted(noun for noun,group in groups.items() if group==fold),transforms={'language':xf,'vision':yf},relations={})
        for ri,relation in enumerate(rels):
            name=relation['name'];val_nouns=[noun for noun in manifest['support'][name] if groups[noun]==(fold+1)%config['folds']]
            test_nouns=[noun for noun in manifest['support'][name] if groups[noun]==fold]
            assert train_nouns[name] and val_nouns and test_nouns
            train_y=endpoints(relation,train_nouns[name]);val=endpoints(relation,val_nouns);ev=endpoints(relation,test_nouns)
            d=mapped[name];image_d=(train_y[:,n//2:,1]-train_y[:,n//2:,0]).mean((0,1));proc_d=text_offsets[name]@procrustes
            scale,curve=select_scale(val,d,config['scales']);image_scale,image_curve=select_scale(val,image_d,config['scales']);proc_scale,proc_curve=select_scale(val,proc_d,config['scales'])
            results={'no_shift':translated_scores(ev,d,0),'text':translated_scores(ev,d,scale),'reverse':translated_scores(ev,-d,scale),
                     'image_reference':translated_scores(ev,image_d,image_scale),'procrustes':translated_scores(ev,proc_d,proc_scale)}
            generator=torch.Generator().manual_seed(config['seed']+fold*100+ri)
            random=F.normalize(torch.randn(config['random_controls'],config['dimensions'],generator=generator,dtype=torch.float64),dim=-1)*d.norm()
            random_scores=[translated_scores(ev,rd,scale) for rd in random]
            results['mean_random']={metric:np.mean([a[metric] for a in random_scores],axis=0) for metric in METRICS}
            wrong=[translated_scores(ev,F.normalize(other,dim=0)*d.norm(),scale) for other_name,other in mapped.items() if other_name!=name]
            results['mean_wrong']={metric:np.mean([a[metric] for a in wrong],axis=0) for metric in METRICS}
            direct=torch.stack([(text(relation['target'],noun)-xm)@w+ym for noun in test_nouns])[:,None,:].expand_as(ev[:,:,0])
            specific=torch.stack([(text(relation['target'],noun)-text(relation['source'],noun))@w for noun in test_nouns])[:,None,:]
            results['direct_target_text']=score_queries(direct,ev)
            results['noun_specific_text']=score_queries(ev[:,:,0]+scale*specific,ev)
            for ni,noun in enumerate(test_nouns):
                per_cell.append(dict(relation=ri,noun=noun,fold=fold,values={condition:{m:np.asarray(results[condition][m][ni],dtype=float).tolist() for m in METRICS} for condition in condition_names}))
                vectors.append(dict(relation=name,noun=noun,fold=fold,text_direction=d.numpy(),image_direction=image_d.numpy(),
                                    source=ev[ni,:,0].numpy(),target=ev[ni,:,1].numpy(),scale=scale))
            saved['relations'][name]=dict(train=train_nouns[name],validation=val_nouns,test=test_nouns,scale=scale,image_scale=image_scale,procrustes_scale=proc_scale,
                                         validation_ap=curve,image_validation_ap=image_curve,procrustes_validation_ap=proc_curve,
                                         direction_norm=float(d.norm()),candidates_per_query=len(test_nouns)*2*(n-1))
            print(f'Fold {fold} {name}: scale={scale}, AP {results["no_shift"]["ap"].mean():.3f} → {results["text"]["ap"].mean():.3f}',flush=True)
        fold_info.append(saved)
    relation_ids=np.array([c['relation'] for c in per_cell]);nouns=sorted({c['noun'] for c in per_cell});noun_ids=np.array([nouns.index(c['noun']) for c in per_cell])
    scores={condition:{metric:np.array([c['values'][condition][metric] for c in per_cell]) for metric in METRICS} for condition in condition_names}
    controls=['no_shift','mean_random','reverse','mean_wrong'];names=['text_minus_'+c for c in controls]+['noun_preservation']
    effects=np.stack([scores['text']['ap']-scores[c]['ap'] for c in controls]+[scores['text']['noun']-scores['no_shift']['noun']])
    boot,cat_boot,rel_boot,noun_weights,asset_draws=clustered_bootstrap(effects,relation_ids,noun_ids,config['seed']+2000,config['bootstrap_draws'],len(rels))
    inference={}
    for i,name in enumerate(names):
        alpha=.05/4;loo=[]
        for ni in range(len(nouns)):
            keep=noun_ids!=ni;loo.append(float(macro(effects[i,keep],relation_ids[keep],len(rels))))
        inference[name]=dict(effect=float(macro(effects[i],relation_ids,len(rels))),ci95=np.quantile(boot[i],[.025,.975]).tolist(),
                             ci_adjusted=np.quantile(boot[i],[alpha/2,1-alpha/2]).tolist() if i<4 else None,
                             one_sided_lower95=float(np.quantile(boot[i],.05)),category_only_ci95=np.quantile(cat_boot[i],[.025,.975]).tolist(),
                             leave_one_noun_out=loo)
    passes=all(inference[name]['ci_adjusted'][0]>0 for name in names[:4]) and inference[names[-1]]['one_sided_lower95']> -config['category_preservation_tolerance']
    per_relation={r['name']:dict(nouns=manifest['support'][r['name']],metrics={condition:{metric:float(a[relation_ids==ri].mean()) for metric,a in stats.items()} for condition,stats in scores.items()},
                               contrasts={name:dict(effect=float(effects[i,relation_ids==ri].mean()),ci95_descriptive=np.quantile(rel_boot[i,:,ri],[.025,.975]).tolist()) for i,name in enumerate(names)}) for ri,r in enumerate(rels)}
    source_paths=['STATE_TRANSFER_PROTOCOL.md','configs/state_transfer.json','state_data.py','mit_states_data.py','state_extract.py','state_analysis.py','semantic_analysis.py','tests/test_state_transfer.py']
    result=dict(manifest_hash=digest(manifest),noun_count=len(nouns),relation_noun_cells=len(per_cell),source_queries=len(per_cell)*n,
                primary_metric='macro relation/noun/query target-state-and-noun average precision',conditions={condition:{metric:float(macro(a,relation_ids,len(rels))) for metric,a in stats.items()} for condition,stats in scores.items()},
                inference=inference,relations=per_relation,folds=fold_info,passes_predeclared_criteria=passes,
                source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths})
    write_json(RUN/'results.json',result);write_json(RUN/'per_query.json',per_cell)
    np.savez_compressed(RUN/'bootstrap.npz',bootstrap=boot,category_only=cat_boot,per_relation=rel_boot,noun_weights=noun_weights,query_draws=asset_draws)
    np.savez_compressed(RUN/'vectors.npz',text_directions=np.stack([v['text_direction'] for v in vectors]),image_directions=np.stack([v['image_direction'] for v in vectors]),
                        source=np.stack([v['source'] for v in vectors]),target=np.stack([v['target'] for v in vectors]),scale=np.array([v['scale'] for v in vectors]))
    print(json.dumps({'conditions':result['conditions'],'inference':inference,'passes_predeclared_criteria':passes},indent=2))


if __name__=='__main__':analyze()
