"""Download pinned extra backbones and reuse audited existing pilot model specs."""
import hashlib
import json
from pathlib import Path
from shared.io import write_json
from .download_vggsounder import download

CHECKPOINTS = [
    ('pythia410','EleutherAI/pythia-410m','9879c9b5f8bea9051dcb0e68dff21493d67e9d4f',
     ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors'],'text','text'),
    ('mae_base','facebook/vit-mae-base','25b184bea5538bf5c4c852c79d221195fdd2778d',
     ['config.json','preprocessor_config.json','model.safetensors'],'vision','hf_vision'),
    ('audiomae','gaunernst/vit_base_patch16_1024_128.audiomae_as2m','86ef4d1348c011e4814bbf51f4acfb3094296532',
     ['config.json','model.safetensors'],'audio','audiomae'),
]


def setup_models(pilot_specs, output):
    """Save absolute local paths, immutable revisions and checksums for six models.

    Existing BLOOMZ, DINOv2 and BEATs paths come from the completed pilot, so old
    weights need not download again. Keep this output local, not in Git.
    """
    output=Path(output).resolve()
    if output.exists(): raise ValueError('Model config exists; preserve its pinned identity')
    existing=json.loads(Path(pilot_specs).read_text())
    models={name:existing[name] for name in ['bloomz-560m','dinov2_base','beats_iter1']}
    assets=output.parent/'assets/expanded'
    for name,repo,revision,files,modality,adapter in CHECKPOINTS:
        folder=assets/repo.split('/')[-1];folder.mkdir(parents=True,exist_ok=True)
        for filename in files:
            download(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',folder/filename)
        origin=folder/'origin.json'
        if not origin.exists(): write_json(origin,{'repository':repo,'revision':revision})
        weights=folder/'model.safetensors'
        models[name]={'modality':modality,'adapter':adapter,'checkpoint':str(folder),
                      'origin_repository':repo,'revision':revision,'eligible_unimodal':True,
                      'offline':True,'pooling':'masked_mean' if modality=='text' else 'cls','layer':-1,
                      'weights_path':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                      'local_file_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir() if path.is_file()},
                      'provenance':('Text-only Pythia base' if name=='pythia410' else
                                    'Original image-only MAE pretraining backbone, no supervised fine-tuning' if name=='mae_base' else
                                    'AudioMAE AudioSet2M self-supervised pretraining, converted to timm; no classifier head or fine-tuning; original-byte equivalence not independently established')}
    write_json(output,models)
    return models
