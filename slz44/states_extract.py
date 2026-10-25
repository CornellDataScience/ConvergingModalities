"""Select MIT-States images and extract text/image features (one layer + per-example q95).

    python states_extract.py --zip path/to/release_dataset.zip --run data/states --device cpu

Resumable: rerun the same command after an interruption.
"""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from states_data import build_manifest, save_chunk, write_json

LANGUAGE = 'bigscience/bloomz-1b7'
VISION = 'timm/vit_base_patch14_dinov2.lvd142m'


def pooled(hidden, mask):
    weights = mask.to(dtype=torch.float32).unsqueeze(-1)
    return (hidden.float() * weights).sum(1) / weights.sum(1)


def extract_language(run, manifest, args):
    from transformers import AutoModel, AutoTokenizer
    folder = run / 'features_language'
    meta = folder / 'metadata.json'
    if meta.exists() and json.loads(meta.read_text()).get('complete'):
        print('Cached language features', flush=True)
        return
    texts = [t['text'] for t in manifest['texts']]
    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(LANGUAGE, torch_dtype=torch.float32).eval().to(args.device)
    started = time.monotonic()
    for start in range(0, len(texts), args.batch_size):
        if (folder / f'{start:05d}.npz').exists():
            continue
        tokens = tokenizer(texts[start:start + args.batch_size], padding=True, return_tensors='pt')
        tokens = {k: v.to(args.device) for k, v in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states
            layers = torch.stack([pooled(h, tokens['attention_mask']).cpu() for h in hidden], dim=1)
        q95 = torch.quantile(layers.abs().flatten(1), .95, dim=1)
        save_chunk(folder, start, layers[:, args.language_layer].numpy(), q95.numpy())
        print(f'language: {min(start + args.batch_size, len(texts))}/{len(texts)}', flush=True)
    write_json(meta, dict(complete=True, model=LANGUAGE, layer=args.language_layer,
                          manifest_hash=manifest['hash'], elapsed_seconds=time.monotonic() - started))
    del model


def extract_vision(run, manifest, args):
    import timm
    from PIL import Image
    from timm.data import resolve_data_config, create_transform
    folder = run / 'features_vision'
    meta = folder / 'metadata.json'
    if meta.exists() and json.loads(meta.read_text()).get('complete'):
        print('Cached vision features', flush=True)
        return
    records = manifest['images']
    model = timm.create_model(VISION.split('/', 1)[1], pretrained=True)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model = model.eval().to(args.device)
    captured = []
    handles = [block.register_forward_hook(lambda m, i, o: captured.append(o[:, 0, :].detach().float().cpu()))
               for block in model.blocks]
    started = time.monotonic()
    with zipfile.ZipFile(args.zip) as zf:
        for start in range(0, len(records), args.vision_batch_size):
            if (folder / f'{start:05d}.npz').exists():
                continue
            batch = records[start:start + args.vision_batch_size]
            images = [transform(Image.open(io.BytesIO(zf.read(r['member']))).convert('RGB')) for r in batch]
            captured.clear()
            with torch.inference_mode():
                model.forward_features(torch.stack(images).to(args.device))
            layers = torch.stack(captured, dim=1)
            q95 = torch.quantile(layers.abs().flatten(1), .95, dim=1)
            save_chunk(folder, start, layers[:, args.vision_layer].numpy(), q95.numpy())
            print(f'vision: {start + len(batch)}/{len(records)} ({time.monotonic() - started:.0f}s)', flush=True)
    for handle in handles:
        handle.remove()
    write_json(meta, dict(complete=True, model=VISION, layer=args.vision_layer,
                          manifest_hash=manifest['hash'], elapsed_seconds=time.monotonic() - started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', type=Path, required=True, help='MIT-States release archive')
    parser.add_argument('--run', type=Path, default=Path('data/states'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16, help='text batch size')
    parser.add_argument('--vision-batch-size', type=int, default=8)
    parser.add_argument('--language-layer', type=int, default=19, help='hidden-state index (0 = embeddings)')
    parser.add_argument('--vision-layer', type=int, default=10, help='block index counted from zero')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--modality', choices=['all', 'language', 'vision'], default='all')
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    path = args.run / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
    else:
        print('Scanning archive and sampling images...', flush=True)
        manifest = build_manifest(args.zip)
        write_json(path, manifest)
    counts = {r: len(v) for r, v in manifest['nouns_for'].items()}
    print(f"{len(manifest['images'])} images, {len(manifest['texts'])} prompts; categories per relation: {counts}", flush=True)
    if args.modality in ('all', 'language'):
        extract_language(args.run, manifest, args)
    if args.modality in ('all', 'vision'):
        extract_vision(args.run, manifest, args)


if __name__ == '__main__':
    main()