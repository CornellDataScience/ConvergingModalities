"""Download a bounded Quick, Draw! subset and render exact RGB interventions."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageDraw
import requests
from experiment import digest, write_json

CONFIG = Path('configs/semantic_quickdraw.json')


def render_mask(strokes, size=224):
    points = np.array([(x, y) for xs, ys in strokes for x, y in zip(xs, ys)], dtype=float)
    if len(points) < 2 or np.ptp(points, axis=0).max() < 5:
        raise ValueError('Degenerate drawing')
    lower, upper = points.min(0), points.max(0)
    scale = 180 / max(upper - lower)
    offset = (np.array([224., 224.]) - (upper - lower) * scale) / 2
    canvas = Image.new('L', (size*3, size*3), 0)
    draw = ImageDraw.Draw(canvas)
    for xs, ys in strokes:
        coords = [tuple(((np.array([x, y])-lower)*scale+offset)*3) for x, y in zip(xs, ys)]
        if len(coords) > 1:
            draw.line(coords, fill=255, width=9, joint='curve')
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def recolor(mask, color, background):
    alpha = np.asarray(mask, dtype=float)[..., None] / 255
    return Image.fromarray(np.round(alpha*np.array(color)+(1-alpha)*background).astype('uint8'))


def fetch(category, pool_size, path):
    if path.exists():
        return json.loads(path.read_text())
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/' + quote(category) + '.ndjson'
    rows, ids = [], set()
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            raw = json.loads(line)
            if not raw['recognized'] or raw['key_id'] in ids:
                continue
            try:
                render_mask(raw['drawing'])
            except ValueError:
                continue
            rows.append({k: raw[k] for k in ['key_id', 'word', 'recognized', 'drawing']})
            ids.add(raw['key_id'])
            if len(rows) == pool_size:
                break
    if len(rows) != pool_size:
        raise ValueError('Insufficient valid recognized drawings for '+category)
    write_json(path, rows)
    return rows


def prepare(stage):
    config = json.loads(CONFIG.read_text()); spec = config[stage]
    run = Path('results/semantic_'+stage); run.mkdir(parents=True, exist_ok=True)
    if (run/'manifest.json').exists():
        saved = json.loads((run/'manifest.json').read_text())
        relevant=lambda value:{k:v for k,v in value.items() if k not in ['pilot','main'] or k==stage}
        if relevant(saved['config']) != relevant(config):
            raise ValueError('Config changed; use a separate versioned output directory')
        print('Existing manifest '+str(run)); return
    freeze = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), config=config, config_hash=digest(config),
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [CONFIG, Path('QUICKDRAW_PROTOCOL.md'), Path('quickdraw_data.py'), Path('semantic_extract.py'), Path('semantic_analysis.py'), Path('tests/test_semantic.py'), Path('pilot_planning.py')]})
    write_json(run/'design_lock.json', freeze)
    snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in freeze['source_hashes']:
        target=snapshot/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source).read_bytes())
    records, sources, ids, masks = [], [], set(), set()
    for split in ['train', 'validation', 'test']:
        for category in spec[split]:
            cache = Path('.cache/quickdraw')/(category+'.json')
            pool = fetch(category, config['recognized_pool'], cache)
            # Category-specific seeds do not depend on category traversal order.
            category_seed = int(hashlib.sha256(category.encode()).hexdigest()[:8], 16) + config['seed']
            choice = np.random.default_rng(category_seed).choice(len(pool), spec['instances'], replace=False)
            selected = [pool[int(i)] for i in choice]
            sources.append(dict(category=category, url='https://storage.googleapis.com/quickdraw_dataset/full/simplified/'+quote(category)+'.ndjson',
                                pool_hash=digest(pool), chosen_pool_indices=choice.tolist(), drawings=selected))
            for instance, source in enumerate(selected):
                if source['key_id'] in ids: raise ValueError('Duplicate drawing ID')
                ids.add(source['key_id']); mask = render_mask(source['drawing'])
                mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()
                if mask_hash in masks: raise ValueError('Duplicate mask')
                masks.add(mask_hash)
                background = 235 if instance % 2 == 0 else 245
                pair = []
                for color, rgb in config['colors'].items():
                    image = recolor(mask, rgb, background)
                    image_path = Path('images')/f'{category}_{instance:02d}_{color}.png'
                    (run/image_path).parent.mkdir(parents=True, exist_ok=True);image.save(run/image_path)
                    pair.append(np.asarray(image))
                    template = instance % len(config['templates'])
                    records.append(dict(category=category, instance=instance, asset_id=str(source['key_id']), split=split,
                                        color=color, text=config['templates'][template].format(color=color, category=category),
                                        image=str(image_path), image_sha256=hashlib.sha256((run/image_path).read_bytes()).hexdigest(),
                                        mask_sha256=mask_hash, background=background, template=template,
                                        verification='game-recognized; automated mask audit; no human relabeling',
                                        role='map' if split=='train' and instance<spec['instances']//2 else 'relation' if split=='train' else split))
                changed=np.any(pair[0]!=pair[1],axis=-1); mask_array=np.asarray(mask)
                assert changed.any() and not changed[mask_array==0].any()
            print(f'{stage}: {category} · {len(selected)} drawings',flush=True)
    baseline = json.loads(Path('results/baseline/manifest.json').read_text())
    models = [m for m in baseline['config']['models'] if m['id'] in [config['primary_language'], config['primary_vision']]]
    manifest = dict(stage=stage, config=config, config_hash=digest(config), design_lock_hash=digest(freeze),
                    records=records, sample_hash=digest(records), models=models,
                    attribution='Quick, Draw! dataset © Google, Inc., CC BY 4.0. Recentered and recolored derivatives.',
                    source='https://github.com/googlecreativelab/quickdraw-dataset', license='https://creativecommons.org/licenses/by/4.0/')
    write_json(run/'sources.json', sources);write_json(run/'manifest.json', manifest)
    # Every drawing appears once (red) in contact sheets, six categories per sheet.
    cats=[c for split in ['train','validation','test'] for c in spec[split]]
    for start in range(0,len(cats),6):
        per_category=30+((spec['instances']+9)//10)*82
        sheet=Image.new('RGB',(1000,per_category*len(cats[start:start+6])), 'white');d=ImageDraw.Draw(sheet)
        for k,cat in enumerate(cats[start:start+6]):
            d.text((8,k*per_category+4),cat,fill='black')
            for i in range(spec['instances']):
                image=Image.open(run/'images'/f'{cat}_{i:02d}_red.png').resize((78,78))
                sheet.paste(image,((i%10)*100,k*per_category+25+(i//10)*82))
        sheet.save(run/f'contact_{start//6}.png')
    print(f'Saved {len(records)} images, {len(ids)} drawing IDs; all pair masks verified.',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','main'],required=True)
    prepare(p.parse_args().stage)
