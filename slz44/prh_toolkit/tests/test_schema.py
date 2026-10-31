import json
import math
import pytest
import prh
from prh.__main__ import main as cli
from convert_slz44 import categories, scale_grid, state_matrix


@pytest.fixture
def protocol(tmp_path):
    p = tmp_path / 'PROTOCOL.md'
    p.write_text('plan')
    return p


def make(protocol):
    return prh.new_run('demo-run', '1', protocol, dataset={'name': 'toy', 'n_pool': 10},
                       models=[{'id': 'a', 'modality': 'language'}, {'id': 'b', 'modality': 'vision'}],
                       commit='abc1234')


def test_valid_document_round_trips(tmp_path, protocol):
    doc = make(protocol)
    prh.add_record(doc, 'cka_excess', 'center', 0.25, n=128, seed=1, layers={'language': 3, 'vision': 4},
                   interval_low=0.1, interval_high=0.4, extra={'note': 'x'})
    path = prh.write(doc, tmp_path / 'out' / 'r.json')
    assert prh.read(path) == doc
    assert doc['protocol']['sha256'] == prh.schema.file_sha256(protocol)


def test_bad_records_are_rejected(protocol):
    doc = make(protocol)
    for bad in [dict(value=float('nan')), dict(value=True), dict(n=1.5), dict(typo=1),
                dict(interval_low=2.0, interval_high=1.0), dict(layers={'a': 'x'})]:
        fields = dict(metric='m', condition='c', value=1.0)
        fields.update(bad)
        with pytest.raises(prh.SchemaError):
            prh.add_record(doc, **fields)
    assert doc['records'] == []


def test_bad_documents_report_problems(protocol):
    doc = make(protocol)
    assert prh.validate(doc) == []
    doc['track'] = '9'
    doc['git_commit'] = 'zzz'
    doc['protocol']['sha256'] = 'abc'
    doc['models'][0]['modality'] = 'smell'
    assert len(prh.validate(doc)) == 4
    assert prh.validate({}) != []
    with pytest.raises(prh.SchemaError):
        prh.write(doc, 'unused.json')


def test_cli_validate(tmp_path, protocol, capsys):
    good = prh.write(make(protocol), tmp_path / 'good.json')
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'run_id': 'x'}))
    assert cli(['validate', str(good)]) == 0
    assert cli(['validate', str(good), str(bad)]) == 1
    assert 'problem' in capsys.readouterr().out


def test_converters_produce_valid_documents(protocol):
    rows = [dict(mode='center', n=128, seed=7, language='L', vision='V', knn_fixed=0.1, knn_scaled=0.08,
                 cka=0.2, gw=0.3, k_scaled=2)]
    doc = scale_grid(dict(rows=rows), protocol, commit='abc1234')
    assert prh.validate(doc) == [] and len(doc['records']) == 4
    cats = [dict(condition='reference', fold='0', relation='cooking', noun='beef', cos_own='0.4',
                 margin='0.2', top1='1.0')]
    doc = categories(cats, protocol, commit='abc1234')
    assert prh.validate(doc) == [] and doc['records'][0]['relation'] == 'cooking'
    assert all(math.isfinite(r['value']) for r in doc['records'])


def test_state_matrix_converter(protocol):
    results = dict(summary={'reference': dict(matrix_text=[[0.5 if i == j else 0.1 for j in range(5)] for i in range(5)])})
    doc = state_matrix(results, protocol, commit='abc1234')
    assert prh.validate(doc) == [] and len(doc['records']) == 25
    assert doc['records'][0]['extra']['visual_relation'] == 'cleaning'
