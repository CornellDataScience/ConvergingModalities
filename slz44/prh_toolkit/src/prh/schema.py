"""Shared results format, version 0.1. One JSON file per run; see docs/RESULTS_SCHEMA.md."""
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = '0.1'
TRACKS = ('1', '2', 'shared')
MODALITIES = ('language', 'vision', 'audio', 'other')
RUN_ID = re.compile(r'^[a-z0-9][a-z0-9._-]{2,63}$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
COMMIT = re.compile(r'^([0-9a-f]{7,40}|uncommitted)$')
REQUIRED_RUN = ('schema_version', 'run_id', 'track', 'created', 'git_commit', 'protocol', 'dataset', 'models',
                'records')
REQUIRED_RECORD = ('metric', 'condition', 'value')
STR_FIELDS = ('split', 'relation', 'noun', 'language_model', 'vision_model')
INT_FIELDS = ('n', 'k', 'fold', 'seed')
NUM_FIELDS = ('null_mean', 'null_q95', 'interval_low', 'interval_high')
RECORD_FIELDS = set(REQUIRED_RECORD) | set(STR_FIELDS) | set(INT_FIELDS) | set(NUM_FIELDS) | {'layers', 'extra'}


class SchemaError(ValueError):
    """Raised when a results document does not follow the schema."""


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(cwd=None):
    """Short hash of HEAD, or 'uncommitted' when git is unavailable."""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=cwd, capture_output=True, text=True,
                             check=True)
        return out.stdout.strip() or 'uncommitted'
    except (OSError, subprocess.CalledProcessError):
        return 'uncommitted'


def new_run(run_id, track, protocol_file, dataset, models, notes=None, commit=None, created=None):
    """Start a results document. `protocol_file` is hashed so a run is tied to the plan it followed."""
    protocol_file = Path(protocol_file)
    doc = dict(schema_version=SCHEMA_VERSION, run_id=run_id, track=str(track),
               created=created or datetime.now(timezone.utc).isoformat(timespec='seconds'),
               git_commit=commit or git_commit(protocol_file.resolve().parent),
               protocol=dict(file=protocol_file.name, sha256=file_sha256(protocol_file)),
               dataset=dict(dataset), models=[dict(m) for m in models], records=[])
    if notes:
        doc['notes'] = notes
    return doc


def add_record(doc, metric, condition, value, **fields):
    record = dict(metric=metric, condition=condition, value=value, **fields)
    errors = _check_record(record, 'record')
    if errors:
        raise SchemaError('; '.join(errors))
    doc['records'].append(record)
    return record


def _check_record(r, where):
    if not isinstance(r, dict):
        return [f'{where}: must be an object']
    errors = [f'{where}: missing "{k}"' for k in REQUIRED_RECORD if k not in r]
    errors += [f'{where}: unknown field "{k}" (put extras under "extra")' for k in r if k not in RECORD_FIELDS]
    for k in ('metric', 'condition') + STR_FIELDS:
        if k in r and not (isinstance(r[k], str) and r[k]):
            errors.append(f'{where}: "{k}" must be a non-empty string')
    for k in INT_FIELDS:
        if k in r and not _integer(r[k]):
            errors.append(f'{where}: "{k}" must be an integer')
    for k in ('value',) + NUM_FIELDS:
        if k in r and not _number(r[k]):
            errors.append(f'{where}: "{k}" must be a finite number')
    if _number(r.get('interval_low')) and _number(r.get('interval_high')) and r['interval_low'] > r['interval_high']:
        errors.append(f'{where}: interval_low is above interval_high')
    if 'layers' in r:
        layers = r['layers']
        if not (isinstance(layers, dict) and layers and all(isinstance(k, str) and _integer(v)
                                                              for k, v in layers.items())):
            errors.append(f'{where}: "layers" must map names to integers')
    if 'extra' in r and not isinstance(r['extra'], dict):
        errors.append(f'{where}: "extra" must be an object')
    return errors


def validate(doc):
    """Return a list of problems (empty when the document is valid)."""
    if not isinstance(doc, dict):
        return ['document must be a JSON object']
    errors = [f'missing "{k}"' for k in REQUIRED_RUN if k not in doc]
    if errors:
        return errors
    if doc['schema_version'] != SCHEMA_VERSION:
        errors.append(f'schema_version is {doc["schema_version"]!r}, expected {SCHEMA_VERSION!r}')
    if not (isinstance(doc['run_id'], str) and RUN_ID.match(doc['run_id'])):
        errors.append('run_id must be 3-64 lowercase letters, digits, ".", "_" or "-"')
    if doc['track'] not in TRACKS:
        errors.append(f'track must be one of {TRACKS}')
    try:
        datetime.fromisoformat(str(doc['created']))
    except ValueError:
        errors.append('created must be an ISO 8601 timestamp')
    if not (isinstance(doc['git_commit'], str) and COMMIT.match(doc['git_commit'])):
        errors.append('git_commit must be a 7-40 character hex hash or "uncommitted"')
    protocol = doc['protocol']
    if not (isinstance(protocol, dict) and isinstance(protocol.get('file'), str)
            and isinstance(protocol.get('sha256'), str) and SHA256.match(protocol['sha256'])):
        errors.append('protocol needs "file" and a 64-character hex "sha256"')
    dataset = doc['dataset']
    if not (isinstance(dataset, dict) and isinstance(dataset.get('name'), str) and dataset['name']):
        errors.append('dataset needs a "name"')
    elif 'n_pool' in dataset and not (_integer(dataset['n_pool']) and dataset['n_pool'] > 0):
        errors.append('dataset.n_pool must be a positive integer')
    models = doc['models']
    if not (isinstance(models, list) and models):
        errors.append('models must be a non-empty list')
    else:
        for i, m in enumerate(models):
            if not (isinstance(m, dict) and isinstance(m.get('id'), str) and m.get('modality') in MODALITIES):
                errors.append(f'models[{i}] needs "id" and "modality" in {MODALITIES}')
    if not isinstance(doc['records'], list):
        errors.append('records must be a list')
    else:
        for i, r in enumerate(doc['records'][:10000]):
            errors += _check_record(r, f'records[{i}]')
            if len(errors) > 50:
                errors.append('... further errors omitted')
                break
    return errors


def write(doc, path):
    errors = validate(doc)
    if errors:
        raise SchemaError('\n'.join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, allow_nan=False), encoding='utf-8')
    return path


def read(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    errors = validate(doc)
    if errors:
        raise SchemaError(f'{path}:\n' + '\n'.join(errors))
    return doc
