"""Build the cached baseline features and held-out layer selection that anisotropy.py reads.

Independent implementation of the published measurement recipe (WIT-1024 pairs, BLOOMZ and
DINOv2, per-layer features, 95th-percentile clipping, L2 normalization, k=10, held-out layer
selection on 512/512 splits). No code is shared with any other repository.

    python prepare_data.py extract --out data/baseline --device cpu
    python prepare_data.py heldout --baseline data/baseline --out data/heldout_layers/results.json

Numbers can differ slightly from other implementations (different hardware/precision).
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from anisotropy import digest, load_features, neighbors, overlap, write_json

DATASET = 'minhuh/prh'
REVISION = 'wit_1024'   # branch of the dataset repo; resolved to a commit sha on first run
MODELS = [dict(id='bigscience/bloomz-560m', modality='language'),
          dict(id='bigscience/bloomz-1b1', modality='language'),
          dict(id='bigscience/bloomz-1b7', modality='language'),
          dict(id='timm/vit_small_patch14_dinov2.lvd142m', modality='vision'),
          dict(id='timm/vit_base_patch14_dinov2.lvd142m', modality='vision')]
K = 10
CLIP_QUANTILE = 0.95
SEEDS = [20261004, 20261005, 20261006]
PERMUTATIONS = 1000


def feature_dir(run, spec):
    return run / 'features' / spec['id'].replace('/', '__')


def write_chunk(target, start, features):
    target.mkdir(parents=True, exist_ok=True)
    output = target / f'{start:04d}.npz'
    temp = output.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, feats=np.asarray(features, dtype=np.float32),
                 rows=np.arange(start, start + len(features)))
    temp.replace(output)


def pooled(hidden, mask):
    weights = mask.to(dtype=torch.float32).unsqueeze(-1)
    return (hidden.float() * weights).sum(1) / weights.sum(1)


def prepare_manifest(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    path = args.out / 'manifest.json'
    api = HfApi()
    if path.exists():
        manifest = json.loads(path.read_text())
        revision = manifest['config']['dataset_revision']
    else:
        try:
            revision = api.dataset_info(DATASET, revision=REVISION).sha
        except Exception as error:
            branches = [b.name for b in api.list_repo_refs(DATASET, repo_type='dataset').branches]
            raise SystemExit(f'Could not resolve {DATASET}@{REVISION}: {error}\nBranches: {branches}')
    data = load_dataset(DATASET, revision=revision, split='train')
    if path.exists():
        return data.select(range(manifest['samples'])), manifest
    if len(data) < args.samples:
        raise SystemExit(f'Dataset has only {len(data)} rows')
    data = data.select(range(args.samples))
    records = []
    for i, row in enumerate(data):
        captions = row['text']
        text = captions[0] if isinstance(captions, (list, tuple)) else captions
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Invalid caption at row {i}')
        records.append(dict(source_row=i, text=text,
                            image_sha256=hashlib.sha256(row['image'].convert('RGB').tobytes()).hexdigest()))
    models = [dict(m, revision=api.model_info(m['id']).sha) for m in MODELS]
    manifest = dict(config=dict(dataset=DATASET, dataset_revision=revision, models=models, k=K,
                                clip_quantile=CLIP_QUANTILE, dtype='float32', caption_index=0),
                    records=records, sample_hash=digest(records), samples=len(records))
    write_json(path, manifest)
    return data, manifest


def extract(args):
    import timm
    from timm.data import resolve_data_config, create_transform
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(args.threads)
    data, manifest = prepare_manifest(args)
    print(f"Manifest ready: {manifest['samples']} pairs", flush=True)
    chunks = list(range(0, manifest['samples'], args.batch_size))
    for spec in manifest['config']['models']:
        target = feature_dir(args.out, spec)
        meta_path = target / 'metadata.json'
        identity = dict(model=spec, sample_hash=manifest['sample_hash'], dtype='float32',
                        batch_size=args.batch_size, device=args.device)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['identity'] != identity:
                raise ValueError('Extraction settings differ from cache; use another --out')
            if meta['complete']:
                print('Cached: ' + spec['id'], flush=True)
                continue
        else:
            write_json(meta_path, dict(identity=identity, complete=False))
        print(f"Loading {spec['id']} on {args.device}", flush=True)
        started = time.monotonic()
        hooks, captured = [], []
        if spec['modality'] == 'language':
            tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'])
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            model = AutoModel.from_pretrained(spec['id'], revision=spec['revision'],
                                              torch_dtype=torch.float32).eval().to(args.device)
        else:
            model = timm.create_model(spec['id'].split('/', 1)[1], pretrained=True)
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            model = model.eval().to(args.device)
            hooks = [block.register_forward_hook(lambda m, i, o: captured.append(o[:, 0, :].detach().float().cpu()))
                     for block in model.blocks]
        for start in chunks:
            if (target / f'{start:04d}.npz').exists():
                continue
            rows = data[start:start + args.batch_size]
            with torch.inference_mode():
                if spec['modality'] == 'language':
                    tokens = tokenizer(rows['text'] if isinstance(rows['text'][0], str) else [t[0] for t in rows['text']],
                                       padding=True, return_tensors='pt')
                    tokens = {k: v.to(args.device) for k, v in tokens.items()}
                    result = model(**tokens, output_hidden_states=True, use_cache=False)
                    features = torch.stack([pooled(h, tokens['attention_mask']).cpu()
                                            for h in result.hidden_states], dim=1)
                else:
                    captured.clear()
                    batch = torch.stack([transform(image.convert('RGB')) for image in rows['image']])
                    model.forward_features(batch.to(args.device))
                    features = torch.stack(captured, dim=1)
            if not torch.isfinite(features).all():
                raise ValueError('Nonfinite model features')
            write_chunk(target, start, features.numpy())
            print(f"{spec['id']}: {start + len(features)}/{manifest['samples']}", flush=True)
        for handle in hooks:
            handle.remove()
        with np.load(target / f'{chunks[0]:04d}.npz') as first:
            shape = list(first['feats'].shape[1:])
        write_json(meta_path, dict(identity=identity, complete=True, feature_shape=shape,
                                   elapsed_seconds=time.monotonic() - started))
        del model
        gc.collect()


def fit_threshold(raw, rows, q):
    """Mean over samples of each sample's q-quantile of |features| (chunked for memory)."""
    flat = raw[rows].abs().flatten(1)
    return torch.cat([torch.quantile(flat[i:i + 64], q, dim=1) for i in range(0, len(flat), 64)]).mean()


def layer_matrix(left, right):
    """Mean neighbor overlap for every (left layer, right layer) pair."""
    rows = np.arange(len(left[0]))[:, None]
    result = np.empty((len(left), len(right)))
    for j, nb in enumerate(right):
        membership = np.zeros((len(nb), len(nb)), dtype=bool)
        membership[rows, nb] = True
        for i, na in enumerate(left):
            result[i, j] = membership[rows, na].mean()
    return result


def heldout(args):
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    config = manifest['config']
    n = manifest['samples']
    splits = []
    for seed in SEEDS:
        p = np.random.default_rng(seed).permutation(n)
        splits.append(dict(seed=seed, selection=p[:n // 2].tolist(), evaluation=p[n // 2:].tolist()))
    graphs, thresholds = {}, {}
    for spec in config['models']:
        name = spec['id']
        raw = load_features(args.baseline, spec, manifest)
        graphs[name], thresholds[name] = [], []
        for split in splits:
            threshold = fit_threshold(raw, split['selection'], config['clip_quantile'])
            thresholds[name].append(float(threshold))
            sides = {}
            for side in ['selection', 'evaluation']:
                x = F.normalize(raw[split[side]].clamp(-threshold, threshold), dim=-1)
                sides[side] = [neighbors(x[:, layer], config['k']) for layer in range(x.shape[1])]
            graphs[name].append(sides)
        del raw
        print('Prepared ' + name, flush=True)
    results = []
    for s, split in enumerate(splits):
        for language in [m['id'] for m in config['models'] if m['modality'] == 'language']:
            for vision in [m['id'] for m in config['models'] if m['modality'] == 'vision']:
                a, b = graphs[language][s], graphs[vision][s]
                train = layer_matrix(a['selection'], b['selection'])
                i, j = np.unravel_index(train.argmax(), train.shape)
                evaluation = layer_matrix(a['evaluation'], b['evaluation'])
                left, right = a['evaluation'][i], b['evaluation'][j]
                observed = overlap(left, right)
                assert np.isclose(observed, evaluation[i, j])
                rng = np.random.default_rng(split['seed'] + 10000)
                null = []
                for _ in range(PERMUTATIONS):
                    p = rng.permutation(len(right))
                    null.append(overlap(left, np.argsort(p)[right[p]]))
                results.append(dict(seed=split['seed'], language=language, vision=vision,
                                    layers=[int(i), int(j)], selection_score=float(train[i, j]),
                                    evaluation_score=observed, evaluation_max=float(evaluation.max()),
                                    null_mean=float(np.mean(null)), null_q95=float(np.quantile(null, .95)),
                                    excess=observed - float(np.mean(null)),
                                    p=(1 + sum(v >= observed for v in null)) / (1 + PERMUTATIONS), null=null))
                print(f"{split['seed']} {language} {vision}: {observed:.6f}", flush=True)
    write_json(args.out, dict(manifest_hash=digest(manifest), splits=splits, thresholds=thresholds,
                              k=config['k'], permutations=PERMUTATIONS, results=results))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract', help='download data/models and cache per-layer features')
    e.add_argument('--out', type=Path, default=Path('data/baseline'))
    e.add_argument('--samples', type=int, default=1024)
    e.add_argument('--batch-size', type=int, default=4)
    e.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    e.add_argument('--threads', type=int, default=4)
    h = sub.add_parser('heldout', help='select layers on 512 pairs, evaluate on the other 512')
    h.add_argument('--baseline', type=Path, default=Path('data/baseline'))
    h.add_argument('--out', type=Path, default=Path('data/heldout_layers/results.json'))
    h.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    {'extract': extract, 'heldout': heldout}[args.command](args)


if __name__ == '__main__':
    main()