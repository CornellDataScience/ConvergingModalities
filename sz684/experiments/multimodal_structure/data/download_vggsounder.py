"""Download a small, pinned, real audiovisual pilot (never the whole dataset).

Official video-level annotations do not verify the selected midpoint frame.
This is exploratory development data, not a new independently annotated test set.
"""
import csv
import hashlib
import io
import json
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
from shared.io import write_json
from .split import assign_splits
from .prepare import prepare

ANNOTATION_COMMIT = '5a91bf6851d8cbf830d85b00ba0953f1befaffa4'
MEDIA_COMMIT = 'dc7815aa65132c48f070686e0004180f03b5abcf'
ANNOTATION_URL = f'https://raw.githubusercontent.com/Bizilizi/VGGSounder/{ANNOTATION_COMMIT}/vggsounder/data/vggsounder.csv'
MEDIA_ROOT = f'https://huggingface.co/datasets/11hu83/vggsound/resolve/{MEDIA_COMMIT}'


def download(url, path):
    """Write downloads atomically; existing bytes are checked by the final manifest."""
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_pilot(output, concept_count=24, examples=3, seed=42):
    """Sample single-AV-label clips without narration/music; log every failure."""
    import imageio_ffmpeg
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Pilot is already prepared; use a new run folder')
    annotation = output/'annotations.csv'
    annotation_hash = download(ANNOTATION_URL, annotation)
    clips = defaultdict(list)
    with annotation.open(newline='') as handle:
        for row in csv.DictReader(handle):
            clips[row['video_id']].append(row)
    candidates = defaultdict(list)
    for clip_id, labels in clips.items():
        if any(row[key].lower() == 'true' for row in labels
               for key in ['voice_over','background_music','static_image']):
            continue
        visible_audible = [row['label'] for row in labels if row['modality'] == 'AV']
        if len(visible_audible) == 1:
            candidates[visible_audible[0]].append(clip_id)
    eligible = sorted(label for label, ids in candidates.items() if len(ids) >= examples)
    if len(eligible) < concept_count:
        raise ValueError('Not enough supported concepts under the fixed filters')
    rng = np.random.default_rng(seed)
    labels = rng.choice(eligible, concept_count, replace=False).tolist()
    rows, failures, downloads = [], [], []
    for label in labels:
        successful = 0
        for clip_id in rng.permutation(sorted(candidates[label])):
            if successful == examples:
                break
            video = output/(clip_id+'.mp4')
            audio = output/(clip_id+'.wav')
            image = output/(clip_id+'.jpg')
            try:
                hashes = {'video': download(f'{MEDIA_ROOT}/video/{clip_id}/video.mp4', video),
                          'audio': download(f'{MEDIA_ROOT}/audio/{clip_id}/audio.wav', audio)}
                if not image.exists():
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-nostdin','-n',
                                    '-ss','5','-i',str(video),'-frames:v','1',str(image)], check=True)
                if not image.is_file():
                    raise ValueError('No midpoint frame decoded')
            except Exception as error:
                failures.append({'clip_id': str(clip_id), 'reason': str(error)})
                continue
            downloads.append({'clip_id': str(clip_id), 'label': label, 'sha256': hashes})
            for modality, path in [('vision',image),('audio',audio)]:
                rows.append({'sample_id':str(clip_id)+'_'+modality, 'concept_id':label,
                             'family_id':label, 'source_id':str(clip_id).rsplit('_',1)[0],
                             'modality':modality, 'path':path.name, 'text':''})
            successful += 1
            print(f'{label}: {successful}/{examples}', flush=True)
        if successful < examples:
            failures.append({'label':label,'reason':'Insufficient downloadable examples'})
    # Text names come from human labels; no text enters a vision/audio encoder.
    for i, label in enumerate(labels):
        if any(row['concept_id'] == label for row in rows):
            for j, template in enumerate(['{}', 'the sound of {}', 'an example of {}']):
                rows.append({'sample_id':f'text_{i}_{j}', 'concept_id':label, 'family_id':label,
                             'source_id':f'label_{i}_{j}', 'modality':'text','path':'',
                             'text':template.format(label)})
    rows = assign_splits(rows, seed)
    fields = ['sample_id','concept_id','family_id','source_id','split','modality','path','text']
    with (output/'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'download_log.json', {'annotation_commit':ANNOTATION_COMMIT,
               'media_commit':MEDIA_COMMIT,'annotation_sha256':annotation_hash,
               'seed':seed,'selected_labels':labels,'examples_per_label':examples,
               'downloads':downloads,'failures':failures})
    manifest = prepare(output/'records.csv', output/'manifest.json')
    manifest['status'] = ('EXPLORATORY VGGSounder video-level AV labels; midpoint frames not independently verified; '
                          'family grouping uses exact event labels, not a reviewed taxonomy')
    write_json(output/'manifest.json', manifest)
    return manifest
