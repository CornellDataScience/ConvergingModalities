"""Pre-extraction media QA; these checks do not claim human label verification."""
import hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image
from shared.io import write_json
from .prepare import load_manifest
from .audio import load_audio


def audit_media(manifest_path, output):
    """Decode inputs and record source leakage, exact duplicates and flat frames."""
    manifest=load_manifest(manifest_path)
    records=manifest['records']
    pixels=defaultdict(list);failures=[];flat=[];sources=defaultdict(set)
    for row in records:
        sources[row['source_id']].add(row['split'])
        try:
            if row['modality']=='vision':
                with Image.open(row['path']) as image:
                    image=image.convert('RGB')
                    array=np.asarray(image)
                    key=hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
                    pixels[key].append({'sample_id':row['sample_id'],'split':row['split']})
                    if float(array.std())<2:
                        flat.append(row['sample_id'])
            elif row['modality']=='audio':
                waveform=load_audio(row['path'])
                if len(waveform)<1600: raise ValueError('Less than 0.1 seconds of sound')
        except Exception as error:
            failures.append({'sample_id':row['sample_id'],'error':str(error)})
    duplicate_groups=[values for values in pixels.values() if len(values)>1]
    crossing=[values for values in duplicate_groups if len({row['split'] for row in values})>1]
    result={'dataset_hash':manifest['sample_hash'],'records':len(records),
            'by_modality':dict(Counter(row['modality'] for row in records)),
            'concepts_by_split':{split:len({row['concept_id'] for row in records if row['split']==split})
                                 for split in ['train','validation','test']},
            'decode_failures':failures,'flat_frame_ids':flat,'exact_pixel_duplicate_groups':duplicate_groups,
            'cross_split_pixel_duplicates':crossing,
            'source_ids_crossing_splits':[source for source,splits in sources.items() if len(splits)>1],
            'independent_label_verification':False}
    write_json(Path(output),result)
    if failures or crossing or result['source_ids_crossing_splits']:
        raise ValueError('Media QA failed; inspect the saved audit before extraction')
    return result


def replace_unusable_audio(study_folder):
    """Replace only undecodable/silent recordings, using the saved backup order.

    This must run before feature extraction. It never consults model scores.
    Original manifests and the reason for every substitution are preserved.
    """
    import csv
    import json
    from .expanded import fetch_concept
    from .prepare import prepare
    study=Path(study_folder)
    if (study/'features').exists(): raise ValueError('Media must be finalized before extraction')
    audit=json.loads((study/'media_audit.json').read_text())
    manifest=load_manifest(study/'manifest.json')
    plan=json.loads((study/'sampling_plan.json').read_text())
    if audit['cross_split_pixel_duplicates'] or audit['source_ids_crossing_splits']:
        raise ValueError('This helper cannot repair source/duplicate split leakage')
    index={row['sample_id']:row for row in manifest['records']}
    used={row['source_id'] for row in manifest['records']}
    replacements=[]
    for failure in audit['decode_failures']:
        row=index[failure['sample_id']]
        if row['modality']!='audio': raise ValueError('Only audio failures can use this helper')
        old_clip=row['sample_id'].rsplit('_',1)[0]
        for clip in plan['candidate_order'][row['concept_id']]:
            source=clip.rsplit('_',1)[0]
            if source in used: continue
            candidate_plan=dict(plan,examples_per_concept=1,candidate_order={row['concept_id']:[clip]})
            candidates,_=fetch_concept(row['concept_id'],candidate_plan,study/'media')
            if not candidates: continue
            try:
                load_audio(study/'media'/(clip+'.wav'))
                with Image.open(study/'media'/(clip+'.jpg')) as image: image.verify()
            except Exception:
                used.add(source);continue
            for other in manifest['records']:
                if other['sample_id'] in [old_clip+'_vision',old_clip+'_audio']:
                    suffix='.jpg' if other['modality']=='vision' else '.wav'
                    other['sample_id']=clip+'_'+other['modality']
                    other['source_id']=source
                    other['path']=str((study/'media'/(clip+suffix)).resolve())
                    other.pop('sha256',None)
            used.add(source)
            replacements.append({'old_clip':old_clip,'new_clip':clip,'concept':row['concept_id'],'reason':failure['error']})
            break
        else: raise ValueError('No usable prespecified backup for '+row['concept_id'])
    if not replacements: return manifest
    original=study/'manifest_before_media_qa.json'
    if not original.exists(): write_json(original,load_manifest(study/'manifest.json'))
    write_json(study/'media_replacements.json',replacements)
    fields=['sample_id','concept_id','family_id','source_id','split','modality','path','text']
    with (study/'records.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(manifest['records'])
    updated=prepare(study/'records.csv',study/'manifest.json')
    updated['status']=manifest['status'];updated['sampling_plan_hash']=manifest['sampling_plan_hash']
    updated['pre_extraction_quality_replacements']=len(replacements)
    write_json(study/'manifest.json',updated)
    return updated
