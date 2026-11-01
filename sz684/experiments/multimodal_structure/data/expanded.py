"""Plan and download a larger study before any model sees the new test data.

Sampling, backup order, broad family names and source splits are written first.
The labels are released video-level annotations, not new human verification.
"""
import csv
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from shared.io import digest, write_json
from .download_vggsounder import ANNOTATION_COMMIT, MEDIA_COMMIT, MEDIA_ROOT, download
from .prepare import prepare
from .split import assign_splits
from .axes import covered_splits


def broad_family(label):
    """Readable provisional groups prevent obvious close labels crossing splits.

    These are manual semantic groups, not a validated biological/instrument taxonomy.
    The mapping is frozen in the sampling plan and exported for review.
    """
    instruments = {
        'strings': ['guitar','bass','cello','violin','banjo','ukulele','sitar','harp','mandolin','zither','erhu'],
        'brass': ['trumpet','trombone','horn','bugle','cornet','shofar'],
        'woodwind': ['flute','clarinet','oboe','bassoon','saxophone','harmonica','didgeridoo','bagpipes'],
        'keyboards': ['piano','organ','harpsichord','synthesizer','accordion','theremin'],
        'percussion': ['drum','bongo','conga','cymbal','djembe','gong','guiro','marimba','glockenspiel',
                       'snare','steelpan','tabla','tambourine','timbale','timpani','tympani','vibraphone','washboard','castanet','tuning fork'],
    }
    if label in ['singing bowl','wind chime']: return 'instrument_percussion'
    if label == 'disc scratching': return 'instrument_electronic'
    if label.startswith('playing '):
        for family, words in instruments.items():
            if any(re.search(r'\b'+re.escape(word)+r'\b',label) for word in words):
                return 'instrument_'+family
        return 'sports'
    bird_names = ['bird','oriole','swallow','chickadee','crow','owl','canary','eagle','pigeon',
                  'duck','goose','chicken','turkey','woodpecker','francolin','magpie','penguin',
                  'parrot','pheasant','mynah','warbler','wood thrush']
    if any(re.search(r'\b'+re.escape(name)+r'\b',label) for name in bird_names):
        return 'birds'
    if label.startswith(('people ','male ','female ','baby ','child ','children ')) or label in ['beat boxing','rapping','yodelling','singing choir']:
        return 'human_vocal_and_body'
    if any(word in label for word in ['cat ','lions ','cheetah']): return 'felines'
    if any(word in label for word in ['dog ','fox ','coyote']): return 'canines'
    if any(word in label for word in ['engine','car ','vehicle','bus','motorcycle','snowmobile','tractor']): return 'road_vehicles'
    if any(word in label for word in ['train','railroad','subway']): return 'rail'
    if any(word in label for word in ['airplane','helicopter']): return 'aircraft'
    if any(word in label for word in ['boat','sailing','canoe','kayak']): return 'watercraft'
    if any(word in label for word in ['water','burbling','raining','sea waves','swimming','scuba','toilet']): return 'water'
    if any(word in label for word in ['siren','alarm','bell','beep','horn','detector']): return 'alarms'
    if any(word in label for word in ['fire','gun','cannon','muskets','missile','explosion']): return 'combustion'
    if any(word in label for word in ['wind','thunder','tornado','hail','ice cracking']): return 'weather'
    if any(word in label for word in ['fan','dryer','shaver','razor','blender','ventilation','vacuum','conditioning','printer','keyboard','typewriter','sewing']): return 'appliances'
    if any(word in label for word in ['chopping','hammer','grind','sharpen','welding','lathe','timber','forging','trimmer','mowing','chainsaw']): return 'tools'
    if any(word in label for word in ['door','drawer','cupboard','window']): return 'open_close'
    if any(word in label for word in ['bee','fly','mosquito','cricket','frog','snake']): return 'small_animals'
    return 'other_'+label.split()[0].replace(',','')


def plan_study(annotation_path, pilot_log, output, concepts=80, examples=15, seed=20261023):
    """Freeze candidate order and family/source groups before model extraction."""
    output = Path(output)
    if (output/'sampling_plan.json').exists():
        return json.loads((output/'sampling_plan.json').read_text())
    old = json.loads(Path(pilot_log).read_text())
    excluded_sources = {row['clip_id'].rsplit('_',1)[0] for row in old['downloads']}
    grouped = defaultdict(list)
    with Path(annotation_path).open(newline='') as handle:
        for row in csv.DictReader(handle): grouped[row['video_id']].append(row)
    candidates = defaultdict(list)
    for clip_id, labels in sorted(grouped.items()):
        if clip_id.rsplit('_',1)[0] in excluded_sources: continue
        if any(row[key].lower() == 'true' for row in labels for key in ['voice_over','background_music','static_image']): continue
        av = [row['label'] for row in labels if row['modality']=='AV']
        if len(av)==1 and av[0] not in old['selected_labels']:
            candidates[av[0]].append(clip_id)
    # Pick at most one clip from each original YouTube recording globally.
    unique = defaultdict(list)
    seen = set()
    for label in sorted(candidates):
        for clip in candidates[label]:
            source = clip.rsplit('_',1)[0]
            if source not in seen:
                unique[label].append(clip); seen.add(source)
    eligible = sorted(label for label, clips in unique.items() if len(clips)>=examples)
    if len(eligible)<concepts:
        raise ValueError(f'Only {len(eligible)} concepts meet the declared support and source exclusions')
    rng = np.random.default_rng(seed)
    labels = sorted(rng.choice(eligible,concepts,replace=False).tolist())
    families = {label:broad_family(label) for label in labels}
    dummy = [dict(family_id=families[label],source_id='label_'+label,concept_id=label) for label in labels]
    split_rows = assign_splits(dummy,seed)
    splits = {row['concept_id']:row['split'] for row in split_rows}
    plan = {'schema_version':1,'seed':seed,'annotation_commit':ANNOTATION_COMMIT,'media_commit':MEDIA_COMMIT,
            'examples_per_concept':examples,'labels':labels,'family_by_label':families,'split_by_label':splits,
            'candidate_order':{label:rng.permutation(unique[label]).tolist() for label in labels},
            'excluded_pilot_sources':sorted(excluded_sources),'excluded_pilot_concepts':old['selected_labels'],
            'status':'Prospective released-label study; provisional family grouping; not independent human reannotation'}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'sampling_plan.json',plan)
    return plan


def fetch_concept(label, plan, media):
    """Download in a prespecified backup order; never replace based on model scores."""
    import imageio_ffmpeg
    rows, log = [], []
    for clip in plan['candidate_order'][label]:
        if len(rows)==plan['examples_per_concept']: break
        video, audio, image = media/(clip+'.mp4'), media/(clip+'.wav'), media/(clip+'.jpg')
        try:
            hashes = {'video':download(f'{MEDIA_ROOT}/video/{clip}/video.mp4',video),
                      'audio':download(f'{MEDIA_ROOT}/audio/{clip}/audio.wav',audio)}
            if not image.exists():
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-n',
                                '-ss','5','-i',str(video),'-frames:v','1',str(image)],check=True)
            if not image.is_file(): raise ValueError('No midpoint image decoded')
            rows.append({'clip_id':clip,'label':label,'sha256':hashes})
        except Exception as error:
            log.append({'clip_id':clip,'reason':str(error)})
    return rows, log


def acquire_study(output, workers=4):
    """Parallelize independent media downloads, keeping output ordering fixed."""
    output = Path(output).resolve()
    if (output/'manifest.json').exists(): raise ValueError('Completed study exists; use its saved manifest')
    plan = json.loads((output/'sampling_plan.json').read_text())
    media = output/'media'; media.mkdir(exist_ok=True)
    downloaded, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch_concept,label,plan,media) for label in plan['labels']]
        for label, job in zip(plan['labels'],jobs):
            rows, errors = job.result(); downloaded.extend(rows); failures.extend(errors)
            print(f'{label}: {len(rows)}/{plan["examples_per_concept"]}',flush=True)
            write_json(output/'download_progress.json',{'downloads':downloaded,'failures':failures})
    write_json(output/'download_log.json',{'plan_hash':digest(plan),'downloads':downloaded,'failures':failures})
    counts = {label:sum(row['label']==label for row in downloaded) for label in plan['labels']}
    if any(count!=plan['examples_per_concept'] for count in counts.values()):
        raise ValueError('Insufficient accessible media; no incomplete manifest will be scored')
    records = []
    for row in downloaded:
        clip,label = row['clip_id'],row['label']
        for modality,suffix in [('vision','.jpg'),('audio','.wav')]:
            records.append(dict(sample_id=clip+'_'+modality,concept_id=label,family_id=plan['family_by_label'][label],
                                source_id=clip.rsplit('_',1)[0],split=plan['split_by_label'][label],modality=modality,
                                path=str(media/(clip+suffix)),text=''))
    templates = ['{}','an example of {}','the sound of {}','an event involving {}']
    for i,label in enumerate(plan['labels']):
        for j,template in enumerate(templates):
            records.append(dict(sample_id=f'text_{i}_{j}',concept_id=label,family_id=plan['family_by_label'][label],
                                source_id=f'label_{i}_{j}',split=plan['split_by_label'][label],modality='text',path='',text=template.format(label)))
    fields=['sample_id','concept_id','family_id','source_id','split','modality','path','text']
    with (output/'records.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)
    manifest=prepare(output/'records.csv',output/'manifest.json')
    manifest['status']=plan['status'];manifest['sampling_plan_hash']=digest(plan)
    write_json(output/'manifest.json',manifest)
    return manifest


def finalize_families(output):
    """Correct provisional taxonomy before extraction, preserving the first plan.

    Called only before any feature cache exists. Selected clips never change.
    Original plans remain available, rather than rewriting design history.
    """
    output=Path(output)
    if (output/'features').exists(): raise ValueError('Cannot amend splits after extraction starts')
    plan=json.loads((output/'sampling_plan.json').read_text())
    if 'amendment' in plan: raise ValueError('Family design was already finalized')
    previous=digest(plan)
    if 'amendment' not in plan:
        write_json(output/'sampling_plan_initial.json',plan)
    plan['family_by_label']={label:broad_family(label) for label in plan['labels']}
    dummy=[dict(family_id=plan['family_by_label'][label],source_id='label_'+label,concept_id=label) for label in plan['labels']]
    for attempt in range(10000):
        split_rows=assign_splits(dummy,plan['seed']+attempt)
        if covered_splits(split_rows): break
    else:
        raise ValueError('No source/family split meets predeclared category-axis coverage')
    plan['split_seed']=plan['seed']+attempt
    plan['split_assignment_attempt']=attempt
    plan['split_by_label']={row['concept_id']:row['split'] for row in split_rows}
    plan['amendment']={'before_feature_extraction':True,'previous_plan_hash':previous,
                       'reason':'Review caught substring taxonomy errors (owl inside growling; bass inside bassoon); use word boundaries'}
    write_json(output/'sampling_plan.json',plan)
    rows=list(csv.DictReader((output/'records.csv').open(newline='')))
    for row in rows:
        row['family_id']=plan['family_by_label'][row['concept_id']]
        row['split']=plan['split_by_label'][row['concept_id']]
    with (output/'records.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=prepare(output/'records.csv',output/'manifest.json')
    manifest['status']=plan['status'];manifest['sampling_plan_hash']=digest(plan)
    write_json(output/'manifest.json',manifest)
    return manifest
