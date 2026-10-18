"""End-to-end check on synthetic features: prepare_data.heldout, then anisotropy.main."""
import json
from argparse import Namespace
import numpy as np
from anisotropy import digest, main as run_anisotropy
from prepare_data import feature_dir, heldout, write_chunk


def make_run(run, n=64, layers=3, dim=16):
    rng = np.random.default_rng(0)
    shared = rng.standard_normal((n, 8))
    models = [dict(id=f'bigscience/bloomz-{s}', modality='language') for s in ['560m', '1b1', '1b7']]
    models += [dict(id=f'timm/vit_{s}_patch14_dinov2.lvd142m', modality='vision') for s in ['small', 'base']]
    manifest = dict(config=dict(models=models, k=10, clip_quantile=.95), samples=n, sample_hash='synthetic')
    run.mkdir(parents=True)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    for spec in models:
        mixing = rng.standard_normal((8, dim))
        feats = np.stack([shared @ mixing + 0.5 * rng.standard_normal((n, dim)) + 3. for _ in range(layers)], axis=1)
        target = feature_dir(run, spec)
        write_chunk(target, 0, feats)
        (target / 'metadata.json').write_text(json.dumps(
            dict(identity=dict(sample_hash='synthetic'), complete=True)))
    return manifest


def test_heldout_then_anisotropy(tmp_path):
    run = tmp_path / 'baseline'
    manifest = make_run(run)
    out = tmp_path / 'heldout' / 'results.json'
    heldout(Namespace(baseline=run, out=out, threads=1))
    saved = json.loads(out.read_text())
    assert saved['manifest_hash'] == digest(manifest)
    assert len(saved['results']) == 18
    assert all(len(r['null']) == 1000 for r in saved['results'])
    report = tmp_path / 'anisotropy'
    run_anisotropy(['--baseline', str(run), '--heldout', str(out), '--out', str(report)])
    assert (report / 'REPORT.md').exists() and (report / 'scores.csv').exists()