"""Original image-only MAE backbone through the existing Transformers runtime."""
import torch


def load_model(spec, device):
    """Load pinned local files and disable masking for deterministic full images."""
    from transformers import AutoImageProcessor, ViTMAEModel
    processor = AutoImageProcessor.from_pretrained(spec['checkpoint'], local_files_only=True)
    model = ViTMAEModel.from_pretrained(spec['checkpoint'], local_files_only=True)
    model.config.mask_ratio = 0
    return processor, model.eval().to(device)


def extract_batch(records, loaded, spec, device):
    """Return CLS features; fixed noise makes patch order repeatable."""
    from PIL import Image
    processor, model = loaded
    images = []
    for row in records:
        with Image.open(row['path']) as image:
            images.append(image.convert('RGB').copy())
    pixels = processor(images=images, return_tensors='pt')['pixel_values'].to(device)
    patches = (model.config.image_size // model.config.patch_size)**2
    noise = torch.arange(patches, device=device).float().repeat(len(images), 1)
    with torch.inference_mode():
        tokens = model(pixel_values=pixels, noise=noise).last_hidden_state
    return tokens[:,0].float().cpu().numpy()
