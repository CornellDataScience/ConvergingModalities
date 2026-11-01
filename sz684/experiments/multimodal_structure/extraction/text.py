"""Frozen text-only transformer extraction with masked mean pooling."""
import numpy as np
import torch


def load_model(spec, device):
    """Load exactly the pinned checkpoint; network access must be explicit."""
    from transformers import AutoModel, AutoTokenizer
    options = {'revision': spec['revision'], 'local_files_only': spec.get('offline', True)}
    if spec.get('cache_dir'):
        options['cache_dir'] = spec['cache_dir']
    tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint'], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(spec['checkpoint'], torch_dtype=torch.float32, **options)
    return tokenizer, model.eval().to(device)


def extract_batch(records, loaded, spec, device):
    """Return [batch, width] features at the declared hidden-state index."""
    tokenizer, model = loaded
    tokens = tokenizer([row['text'] for row in records], padding=True, truncation=True,
                       max_length=spec.get('max_tokens', 128), return_tensors='pt')
    tokens = {name: value.to(device) for name, value in tokens.items()}
    with torch.inference_mode():
        output = model(**tokens, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[spec.get('layer', -1)].float()
        mask = tokens['attention_mask'].unsqueeze(-1)
        features = (hidden * mask).sum(1) / mask.sum(1)
    return features.cpu().numpy()
