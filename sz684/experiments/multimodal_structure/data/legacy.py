"""Import existing state features without running or modifying the old study."""
import json
from pathlib import Path
import numpy as np
from shared.io import digest, write_json
from ..extraction.cache import save_features


def import_legacy(source, output):
    """Reuse pinned BLOOMZ/DINOv2 caches as explicitly exploratory pilot data."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing runs are preserved')
    manifest = json.loads((source/'manifest.json').read_text())
    records = []
    arrays = {}
    for old_modality, modality in [('language', 'text'), ('vision', 'vision')]:
        folder = source/('features_'+old_modality)
        metadata = json.loads((folder/'metadata.json').read_text())
        if not metadata['complete'] or metadata['identity']['manifest_hash'] != digest(manifest):
            raise ValueError('Historical cache is incomplete or belongs to another manifest')
        chunks, row_numbers = [], []
        for path in sorted(folder.glob('*.npz')):
            with np.load(path) as batch:
                chunks.append(batch['features'])
                row_numbers.extend(batch['rows'].tolist())
        features = np.concatenate(chunks)
        if row_numbers != list(range(len(features))):
            raise ValueError('Historical cache rows are missing/reordered')
        text_index = {text: i for i, text in enumerate(metadata.get('items') or [])}
        selected = []
        ids = []
        for i, old in enumerate(manifest['records']):
            sample_id = f'{modality}_{i:04d}'
            split = 'test' if old['group'] == 0 else 'validation' if old['group'] == 1 else 'train'
            record = {'sample_id': sample_id, 'concept_id': old['state']+'|'+old['noun'],
                      'family_id': old['noun'], 'source_id': old['image_sha256'],
                      'modality': modality, 'split': split, 'text': old['text'],
                      'path': str(source/old['image'])}
            for relation in manifest['config']['relations']:
                if old['state'] in [relation['source'], relation['target']]:
                    record['relation'] = relation['name']
                    record['label'] = str(int(old['state'] == relation['target']))
            records.append(record)
            ids.append(sample_id)
            selected.append(features[text_index[old['text']]] if modality == 'text' else features[i])
        spec = next(model for model in manifest['models'] if model['modality'] == old_modality)
        arrays[modality] = (np.stack(selected), ids, spec)
    dataset = {'schema_version': 1, 'records': records, 'sample_hash': digest(records),
               'status': 'EXPLORATORY legacy weak labels; repeated text rows are not independent exemplars',
               'legacy_manifest_hash': digest(manifest)}
    write_json(output/'manifest.json', dataset)
    for modality, (features, ids, model) in arrays.items():
        identity = {'dataset_hash': dataset['sample_hash'], 'model': model,
                    'sample_ids': ids, 'extractor_version': 'legacy-import-v1'}
        save_features(output/'features'/f'{modality}.npz', features, ids, identity)
    return dataset
