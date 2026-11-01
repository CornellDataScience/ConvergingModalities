"""Extract one model into checked, resumable batches, then assemble one cache."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import torch
from shared.io import digest, write_json, device_for
from ..data.prepare import load_manifest
from .cache import cache_identity, save_features, load_features
from . import text, vision, audio, hf_vision, audiomae


def extract(manifest_path, model_path, model_name, output, device='cpu', batch_size=8):
    """Resume only identical inputs/settings; never infer a new cache's identity."""
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    manifest = load_manifest(manifest_path)
    spec = json.loads(Path(model_path).read_text())[model_name]
    if not spec.get('eligible_unimodal') or not spec.get('provenance'):
        raise ValueError('Model needs a documented unimodal eligibility audit')
    if not spec.get('revision') or spec['revision'] in ['main', 'master']:
        raise ValueError('Pin an immutable model revision')
    records = [row for row in manifest['records'] if row['modality'] == spec['modality']]
    if not records:
        raise ValueError('Dataset has no rows for this model modality')
    # Verify content before trusting even a fully assembled cache.
    for row in records:
        payload = row['text'].encode() if spec['modality'] == 'text' else Path(row['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('Dataset content changed: ' + row['sample_id'])
    for field in ['weights_path']:
        if spec.get(field):
            actual = hashlib.sha256(Path(spec[field]).read_bytes()).hexdigest()
            if actual != spec.get('weights_sha256'):
                raise ValueError('Model weight SHA256 mismatch')
    for filename, expected in spec.get('local_file_hashes', {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned local model file changed: ' + filename)
    if spec['modality'] == 'audio' and spec.get('adapter', 'beats') == 'beats':
        for filename in ['BEATs.py', 'backbone.py', 'modules.py']:
            actual = hashlib.sha256((Path(spec['source_path'])/filename).read_bytes()).hexdigest()
            if actual != spec.get('source_sha256', {}).get(filename):
                raise ValueError('Pin the SHA256 of every BEATs source file')
    device = device_for(device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    spec = dict(spec, device=device, batch_size=batch_size,
                packages={name: importlib.metadata.version(name) for name in ['torch','numpy','transformers','timm','torchvision','torchaudio']},
                source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')})
    identity = cache_identity(manifest, spec, records)
    output = Path(output)
    if output.exists():
        load_features(output, identity)
        return output
    chunks = output.parent/(output.stem+'_chunks')
    chunks.mkdir(parents=True, exist_ok=True)
    lock = chunks/'identity.json'
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise ValueError('Partial cache settings changed')
    write_json(lock, identity)
    adapters = {'text': text, 'vision': vision, 'beats': audio, 'hf_vision': hf_vision, 'audiomae': audiomae}
    adapter_name = spec.get('adapter', 'beats' if spec['modality']=='audio' else spec['modality'])
    adapter = adapters[adapter_name]
    loaded = None
    matrices = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        batch_identity = dict(identity, sample_ids=[row['sample_id'] for row in batch])
        path = chunks/f'{start:06d}.npz'
        if not path.exists():
            if loaded is None:
                loaded = adapter.load_model(spec, device)
            features = adapter.extract_batch(batch, loaded, spec, device)
            save_features(path, features, batch_identity['sample_ids'], batch_identity)
        matrices.append(load_features(path, batch_identity)[0])
        print(f'{model_name}: {min(start+batch_size, len(records))}/{len(records)}', flush=True)
    save_features(output, np.concatenate(matrices), identity['sample_ids'], identity)
    return output
