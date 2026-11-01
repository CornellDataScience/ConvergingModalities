"""Frozen timm vision transformer extraction, reusing the baseline convention."""
import torch


def load_model(spec, device):
    """Load a local hashed checkpoint and the architecture's image processor."""
    import timm
    from timm.data import resolve_data_config, create_transform
    from safetensors.torch import load_file
    model = timm.create_model(spec['architecture'], pretrained=False)
    path = spec['weights_path']
    state = load_file(path) if path.endswith('.safetensors') else torch.load(path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    processor = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return processor, model.eval().to(device)


def extract_batch(records, loaded, spec, device):
    """Pool a selected transformer block; intermediate hooks are always removed."""
    from PIL import Image
    processor, model = loaded
    images = []
    for row in records:
        with Image.open(row['path']) as image:
            images.append(processor(image.convert('RGB')))
    captured = []
    layer = spec.get('layer', -1)
    handle = model.blocks[layer].register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.inference_mode():
            model.forward_features(torch.stack(images).to(device))
            tokens = captured[0]
            if spec.get('pooling', 'cls') == 'cls':
                features = tokens[:, 0]
            else:
                features = tokens[:, model.num_prefix_tokens:].mean(1)
        return features.float().cpu().numpy()
    finally:
        handle.remove()
