"""Validate a simple CSV and freeze it as a portable JSON dataset manifest."""
import csv
import hashlib
import json
from pathlib import Path
from shared.io import write_json, digest

REQUIRED = ['sample_id', 'concept_id', 'family_id', 'source_id', 'modality', 'split']


def validate_records(records):
    """Reject ambiguous identities and source/family leakage across splits."""
    samples, groups = set(), {}
    concepts = {}
    for record in records:
        if any(not record.get(key) for key in REQUIRED):
            raise ValueError(f'Every record requires {REQUIRED}')
        if record['sample_id'] in samples:
            raise ValueError('Repeated sample_id: ' + record['sample_id'])
        samples.add(record['sample_id'])
        if record['modality'] not in ['text', 'vision', 'audio']:
            raise ValueError('Unknown modality')
        if record['split'] not in ['train', 'validation', 'test']:
            raise ValueError('Split must be train, validation, or test')
        if record['concept_id'] in concepts and concepts[record['concept_id']] != record['family_id']:
            raise ValueError('One concept cannot belong to multiple families')
        concepts[record['concept_id']] = record['family_id']
        for key in ['source_id', 'family_id']:
            group = (key, record[key])
            if group in groups and groups[group] != record['split']:
                raise ValueError(f'{key} crosses splits: {record[key]}')
            groups[group] = record['split']
        if record['modality'] == 'text' and not record.get('text', '').strip():
            raise ValueError('Text records need nonempty text')
        if record['modality'] != 'text' and not record.get('path'):
            raise ValueError('Audio/image records need a local path')
    if not records:
        raise ValueError('Dataset is empty')


def prepare(csv_path, output):
    """Resolve paths relative to the CSV and hash each input before saving."""
    csv_path = Path(csv_path).resolve()
    with csv_path.open(newline='') as handle:
        records = list(csv.DictReader(handle))
    validate_records(records)
    hashes = {}
    for record in records:
        if record['modality'] == 'text':
            payload = record['text'].encode()
        else:
            path = (csv_path.parent / record['path']).resolve()
            if not path.is_file():
                raise ValueError('Missing media: ' + str(path))
            record['path'] = str(path)
            payload = path.read_bytes()
        record['sha256'] = hashlib.sha256(payload).hexdigest()
        content = (record['modality'], record['sha256'])
        if content in hashes and hashes[content] != record['split']:
            raise ValueError('Identical content crosses splits')
        hashes[content] = record['split']
    manifest = {'schema_version': 1, 'records': records, 'sample_hash': digest(records),
                'status': 'User-supplied labels; independent verification not established'}
    write_json(Path(output), manifest)
    return manifest


def load_manifest(path):
    """Check the manifest's row identity before trusting a feature cache."""
    manifest = json.loads(Path(path).read_text())
    validate_records(manifest['records'])
    if digest(manifest['records']) != manifest['sample_hash']:
        raise ValueError('Manifest records changed after preparation')
    return manifest
