"""Build a five-relation natural-image benchmark from the original MIT-States release."""
import concurrent.futures
import hashlib
import io
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from experiment import digest,write_json
from mit_states_data import catalog,download_member

RUN=Path('results/state_transfer')


def make_folds(nouns,support,seed,folds=5):
    rng=np.random.default_rng(seed)
    for attempt in range(100000):
        order=rng.permutation(nouns).tolist();assignment={noun:i%folds for i,noun in enumerate(order)}
        if all(all(any(assignment[n]==f for n in allowed) for f in range(folds)) for allowed in support.values()):
            return assignment,attempt
    raise ValueError('Unable to make coverage-balanced noun groups')


def prepare():
    config=json.loads(Path('configs/state_transfer.json').read_text());RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'manifest.json').exists():
        if json.loads((RUN/'manifest.json').read_text())['config']!=config:raise ValueError('Configuration changed')
        print('Dataset manifest already exists');return
    data=catalog();seen=set();unique=[];removed=[]
    for entry in sorted(data['files'],key=lambda x:x['path']):
        key=(entry['crc32'],entry['size'])
        if key in seen:removed.append(entry['path']);continue
        seen.add(key);unique.append(entry)
    original_counts=Counter((x['state'],x['noun']) for x in data['files']);available=defaultdict(list)
    for entry in unique:available[entry['state'],entry['noun']].append(entry)
    support={};compositions=set()
    for relation in config['relations']:
        a,b=relation['source'],relation['target']
        nouns=sorted(n for s,n in original_counts if s==a and min(original_counts[a,n],original_counts[b,n])>=config['minimum_available_images'] and
                     min(len(available[a,n]),len(available[b,n]))>=config['images_per_composition'])
        support[relation['name']]=nouns
        for n in nouns:compositions.update([(a,n),(b,n)])
    nouns=sorted({n for s,n in compositions});assignment,attempt=make_folds(nouns,support,config['seed'],config['folds'])
    planned=[]
    for state,noun in sorted(compositions):
        entries=available[state,noun];seed=config['seed']+int(hashlib.sha256(f'{state}/{noun}'.encode()).hexdigest()[:8],16)
        indices=np.random.default_rng(seed).choice(len(entries),config['images_per_composition'],replace=False)
        for i,index in enumerate(indices):
            entry=entries[int(index)]
            planned.append(dict(state=state,noun=noun,instance=i,group=assignment[noun],archive=entry,
                                image=f'images/{state}_{noun.replace(" ","-")}_{i:02d}.jpg',
                                text=config['templates'][i%4].format(state=state,noun=noun)))
    lock=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),config=config,archive={k:v for k,v in data.items() if k!='files'},
              support=support,noun_groups=assignment,assignment_attempt=attempt,planned=planned,removed_crc_size_duplicates=removed,
              source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['configs/state_transfer.json','STATE_TRANSFER_PROTOCOL.md','state_data.py','mit_states_data.py']})
    write_json(RUN/'dataset_lock.json',lock)
    (RUN/'images').mkdir(exist_ok=True)
    def fetch(record):
        path=RUN/record['image']
        if not path.exists():
            payload=download_member(record['archive']);path.write_bytes(payload)
        else:payload=path.read_bytes()
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
        record=dict(record);record['image_sha256']=hashlib.sha256(payload).hexdigest();return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records=[]
        for i,r in enumerate(pool.map(fetch,planned)):
            records.append(r)
            if i%100==0:print(f'Images verified: {i+1}/{len(planned)}',flush=True)
    hashes=[r['image_sha256'] for r in records]
    if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate selected image contents after ZIP CRC/size screen')
    baseline=json.loads(Path('results/baseline/manifest.json').read_text())
    models=[m for m in baseline['config']['models'] if m['id'] in [config['primary_language'],config['primary_vision']]]
    manifest=dict(config=config,records=records,models=models,support=support,noun_groups=assignment,
                  dataset_lock_hash=digest(lock),sample_hash=digest(records),source='MIT-States original release (Isola et al.,2015)',
                  note='Weak noun/state image labels; source/target photos are independent instances, not counterfactual pairs.')
    write_json(RUN/'manifest.json',manifest)
    # All selected images can be inspected through category/state contact sheets.
    for start in range(0,len(compositions),12):
        comps=sorted(compositions)[start:start+12];sheet=Image.new('RGB',(8*100,12*110),'white');draw=ImageDraw.Draw(sheet)
        for k,(state,noun) in enumerate(comps):
            draw.text((4,k*110+2),f'{state} {noun}',fill='black')
            for i in range(config['images_per_composition']):
                r=next(r for r in records if r['state']==state and r['noun']==noun and r['instance']==i)
                with Image.open(RUN/r['image']) as image:
                    image=image.convert('RGB');image.thumbnail((96,86));sheet.paste(image,(i*100,k*110+20))
        sheet.save(RUN/f'contact_{start//12:02d}.png')
    print(json.dumps({'images':len(records),'nouns':len(nouns),'support':{k:len(v) for k,v in support.items()},'removed_duplicate_members':len(removed)},indent=2))


if __name__=='__main__':prepare()
