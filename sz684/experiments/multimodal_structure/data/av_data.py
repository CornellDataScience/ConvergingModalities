"""Turn locally accessible audiovisual clips into separately encoded channels.

Input CSV: clip_id,path,concept_id,family_id,split. Supply reviewed concept labels;
this script does not infer whether the named source is visible or audible.
"""
import csv
import subprocess
from pathlib import Path


def import_clips(csv_path, output):
    """Use ffmpeg to save one midpoint frame and up to ten seconds of audio."""
    csv_path, output = Path(csv_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output folder')
    with csv_path.open(newline='') as handle:
        clips = list(csv.DictReader(handle))
    output.mkdir(parents=True)
    rows = []
    for clip in clips:
        clip_id = clip['clip_id']
        if not clip_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in clip_id):
            raise ValueError('clip_id must contain only letters, numbers, underscores or hyphens')
        source = (csv_path.parent/clip['path']).resolve()
        duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                         'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(source)]))
        start = max(0, (duration-10)/2)
        image = output/(clip_id+'.jpg')
        audio = output/(clip_id+'.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(duration/2),
                        '-i', str(source), '-frames:v', '1', str(image)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(start),
                        '-i', str(source), '-t', '10', '-vn', '-ac', '1', '-ar', '16000', str(audio)], check=True)
        for modality, path in [('vision', image), ('audio', audio)]:
            rows.append({'sample_id': clip_id+'_'+modality, 'concept_id': clip['concept_id'],
                         'family_id': clip['family_id'], 'source_id': clip_id,
                         'split': clip['split'], 'modality': modality, 'path': path.name, 'text': ''})
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Repeated clip IDs')
    fields = ['sample_id','concept_id','family_id','source_id','split','modality','path','text']
    with (output/'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output/'records.csv'
