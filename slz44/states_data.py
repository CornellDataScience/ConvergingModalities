"""MIT-States image selection and feature-cache I/O (torch-free).

Selects images for five state-change relations from the MIT-States release archive and
defines the text prompts. Independent implementation of the design described in the
state-transfer manuscript; the sampled images will not be identical to any other sample.
"""
import hashlib
import json
from pathlib import Path
import zipfile
import zlib
import numpy as np

SEED = 20261016
RELATIONS = [dict(name='cleaning', source='dirty', target='clean'),
             dict(name='cooking', source='raw', target='cooked'),
             dict(name='ripening', source='unripe', target='ripe'),
             dict(name='thawing', source='frozen', target='thawed'),
             dict(name='drying', source='wet', target='dry')]
TEMPLATES = ['a photo of a {state} {noun}', 'an image of a {state} {noun}',
             'a {noun} that is {state}', 'a picture showing a {state} {noun}']
PER_STATE = 8
MINIMUM = 10
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def scan_archive(names):
    """Map (state, noun) -> sorted member names, for folders named '<state> <noun>'."""
    states = {r['source'] for r in RELATIONS} | {r['target'] for r in RELATIONS}
    found = {}
    for name in names:
        if '__MACOSX' in name or not name.lower().endswith(IMAGE_SUFFIXES):
            continue
        parts = name.split('/')
        if len(parts) < 2:
            continue
        state, _, noun = parts[-2].partition(' ')
        if state in states and noun:
            found.setdefault((state, noun), []).append(name)
    return {k: sorted(v) for k, v in found.items()}


def select_images(read, found, seed=SEED, per_state=PER_STATE, minimum=MINIMUM):
    """Per (state, noun): seeded shuffle, skip exact duplicates, keep the first `per_state`.

    A folder is eligible only if it has at least `minimum` distinct images.
    `read(member)` returns the image bytes.
    """
    chosen = {}
    for (state, noun), members in sorted(found.items()):
        if len(members) < minimum:
            continue
        rng = np.random.default_rng([seed, zlib.crc32(f'{state} {noun}'.encode())])
        unique, seen = [], set()
        for k in rng.permutation(len(members)):
            sha = hashlib.sha256(read(members[k])).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            unique.append((members[k], sha))
            if len(unique) == minimum:
                break
        if len(unique) >= minimum:
            chosen[(state, noun)] = unique[:per_state]
    return chosen


def build_manifest(archive, seed=SEED):
    with zipfile.ZipFile(archive) as zf:
        found = scan_archive(zf.namelist())
        # Only read folders whose opposite state also exists for the same noun.
        wanted = {(s, n) for r in RELATIONS for (s, n) in found
                  if s in (r['source'], r['target'])
                  and (r['target'] if s == r['source'] else r['source'], n) in found}
        chosen = select_images(zf.read, {k: v for k, v in found.items() if k in wanted}, seed)
    nouns_for = {r['name']: sorted(n for (s, n) in chosen if s == r['source'] and (r['target'], n) in chosen)
                 for r in RELATIONS}
    images, texts = [], []
    for r in RELATIONS:
        for noun in nouns_for[r['name']]:
            for state in (r['source'], r['target']):
                for j, (member, sha) in enumerate(chosen[(state, noun)]):
                    images.append(dict(relation=r['name'], state=state, noun=noun, index=j,
                                       member=member, sha256=sha))
                for t, template in enumerate(TEMPLATES):
                    texts.append(dict(relation=r['name'], state=state, noun=noun, template=t,
                                      text=template.format(state=state, noun=noun)))
    manifest = dict(config=dict(seed=seed, relations=RELATIONS, templates=TEMPLATES,
                                per_state=PER_STATE, minimum=MINIMUM, archive=Path(archive).name),
                    nouns_for=nouns_for, images=images, texts=texts)
    manifest['hash'] = digest(manifest)
    return manifest


def save_chunk(folder, start, features, q95):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{start:05d}.npz'
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f, features=np.asarray(features, dtype=np.float32), q95=np.asarray(q95, dtype=np.float32),
                 rows=np.arange(start, start + len(features)))
    temp.replace(path)


def load_modality(run, modality, expected):
    """Concatenate cached chunks -> (features [N, D], q95 [N]); rows must be 0..expected-1."""
    folder = run / ('features_' + modality)
    metadata = json.loads((folder / 'metadata.json').read_text())
    if not metadata.get('complete'):
        raise ValueError(f'{modality} features incomplete')
    features, q95, rows = [], [], []
    for file in sorted(folder.glob('*.npz')):
        with np.load(file) as chunk:
            features.append(chunk['features'])
            q95.append(chunk['q95'])
            rows.extend(chunk['rows'].tolist())
    if rows != list(range(expected)):
        raise ValueError(f'{modality}: missing, duplicated, or reordered rows')
    return np.concatenate(features).astype(np.float64), np.concatenate(q95).astype(np.float64)