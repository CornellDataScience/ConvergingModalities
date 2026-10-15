"""Pinned offline model extraction, unique text prompts, resumable vision batches."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiment import digest, pooled, write_json, device_for


def extract(stage, modality, device='mps', batch_size=16):
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import timm
    from timm.data import resolve_data_config, create_transform
    torch.set_num_threads(4);torch.manual_seed(20261015)
    run=Path('results/semantic_'+stage);manifest=json.loads((run/'manifest.json').read_text());config=manifest['config']
    spec=next(m for m in manifest['models'] if m['modality']==modality)
    folder=run/('features_'+modality);folder.mkdir(exist_ok=True)
    device=device_for(device)
    texts=sorted({t.format(category=r['category'],color=c) for r in manifest['records']
                  for t in config['templates'] for c in config['colors']})
    items=texts if modality=='language' else manifest['records']
    identity=dict(manifest_hash=digest(manifest),model=spec,device=device,batch_size=batch_size,
                  layer=config[modality+'_layer'],dtype='float32',extractor_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  packages={k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','timm','numpy']})
    meta=folder/'metadata.json'
    if meta.exists() and json.loads(meta.read_text())['identity']!=identity:raise ValueError('Feature cache identity mismatch')
    if meta.exists() and json.loads(meta.read_text()).get('complete'):
        print('Cached '+modality,flush=True);return
    write_json(meta,dict(identity=identity,complete=False,items=texts if modality=='language' else None))
    for r in manifest['records']:
        assert hashlib.sha256((run/r['image']).read_bytes()).hexdigest()==r['image_sha256']
    if modality=='language':
        tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
        tok.padding_side='left';tok.pad_token=tok.eos_token
        model=AutoModel.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,torch_dtype=torch.float32).eval().to(device)
    else:
        model=timm.create_model(spec['id'].split('/',1)[1],pretrained=False)
        model.load_state_dict(load_file(hf_hub_download(spec['id'],'model.safetensors',revision=spec['revision'],local_files_only=True)),strict=True)
        transform=create_transform(**resolve_data_config(model.pretrained_cfg,model=model))
        captured=[]
        handles=[block.register_forward_hook(lambda module,inputs,output:captured.append(output[:,0,:].detach().float().cpu())) for block in model.blocks]
        model=model.eval().to(device)
    started=time.monotonic()
    for start in range(0,len(items),batch_size):
        path=folder/f'{start:05d}.npz'
        if path.exists():continue
        batch=items[start:start+batch_size]
        with torch.inference_mode():
            if modality=='language':
                tokens={k:v.to(device) for k,v in tok(batch,padding=True,return_tensors='pt').items()}
                outputs=model(**tokens,output_hidden_states=True,use_cache=False)
                all_layers=torch.stack([pooled(h,tokens['attention_mask']).cpu() for h in outputs.hidden_states],dim=1)
                del outputs,tokens
            else:
                captured.clear();images=[]
                for r in batch:
                    with Image.open(run/r['image']) as image:images.append(transform(image.convert('RGB')))
                model.forward_features(torch.stack(images).to(device));all_layers=torch.stack(captured,dim=1)
            assert torch.isfinite(all_layers).all()
            q95=torch.quantile(all_layers.abs().flatten(1),.95,dim=1)
            selected=all_layers[:,config[modality+'_layer']].contiguous()
            with path.with_suffix('.tmp').open('wb') as f:np.savez(f,features=selected.numpy(),q95=q95.numpy(),rows=np.arange(start,start+len(batch)))
            path.with_suffix('.tmp').replace(path)
        if start% (batch_size*4)==0 or start+len(batch)==len(items):print(f'{stage} {modality}: {start+len(batch)}/{len(items)} · {time.monotonic()-started:.1f}s',flush=True)
    if modality=='vision':
        for handle in handles:handle.remove()
    write_json(meta,dict(identity=identity,complete=True,items=texts if modality=='language' else None,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','main'],required=True);p.add_argument('--modality',choices=['language','vision'],required=True)
    p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
    extract(a.stage,a.modality,a.device,a.batch_size)
