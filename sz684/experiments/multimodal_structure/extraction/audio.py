"""BEATs pretraining features from the official local implementation.

The upstream code and weight files must be pinned and hashed in the model spec.
Use the iter1 pretraining checkpoint, never a fine-tuned classifier checkpoint.
"""
import sys
import torch
from ..data.audio import load_audio


def load_model(spec, device):
    """Import the audited upstream directory and reject fine-tuned checkpoints."""
    sys.path.insert(0, spec['source_path'])
    from BEATs import BEATs, BEATsConfig
    checkpoint = torch.load(spec['weights_path'], map_location='cpu', weights_only=True)
    config = BEATsConfig(checkpoint['cfg'])
    if config.finetuned_model:
        raise ValueError('Fine-tuned BEATs is excluded from the primary experiment')
    model = BEATs(config)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().to(device)


def extract_batch(records, model, spec, device):
    """Process one unpadded clip at a time and average actual acoustic tokens."""
    features = []
    for row in records:
        waveform = load_audio(row['path'], 16000, spec.get('max_seconds', 10))
        audio = torch.tensor(waveform[None, :], device=device)
        with torch.inference_mode():
            tokens = model.extract_features(audio)[0]
            features.append(tokens.mean(1).float().cpu())
    return torch.cat(features).numpy()
