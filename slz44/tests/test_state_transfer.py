import hashlib
import io
import zipfile
import numpy as np
import pytest
from states_data import RELATIONS, TEMPLATES, build_manifest, load_modality, save_chunk, scan_archive
from state_transfer import (CONDITIONS, METHODS, analyze, apply_condition, assign_groups, bootstrap,
                            diagnostics, fit_condition, macro, prepare, rank_metrics)


def synthetic(n_nouns=30, seed=0, dim_t=80, dim_i=100):
    """Features with a shared per-relation state shift, a common offset, and noise."""
    rng = np.random.default_rng(seed)
    nouns = [f'n{k:02d}' for k in range(n_nouns)]
    nouns_for = {r['name']: [] for r in RELATIONS}
    for k, n in enumerate(nouns):
        for r in (RELATIONS[k % 5], RELATIONS[(k + 1) % 5]):
            nouns_for[r['name']].append(n)
    latent = 20
    u = {n: rng.standard_normal(latent) for n in nouns}
    state_vec = {s: rng.standard_normal(latent) for r in RELATIONS for s in (r['source'], r['target'])}
    shift = {r['name']: 2.0 * rng.standard_normal(latent) for r in RELATIONS}
    at, ai = rng.standard_normal((latent, dim_t)), rng.standard_normal((latent, dim_i))
    offset_t, offset_i = 5 * rng.standard_normal(dim_t), 5 * rng.standard_normal(dim_i)
    images, texts, img, txt = [], [], [], []
    for r in RELATIONS:
        for n in nouns_for[r['name']]:
            for state in (r['source'], r['target']):
                z = u[n] + state_vec[r['source']] + (shift[r['name']] if state == r['target'] else 0)
                for j in range(8):
                    images.append(dict(relation=r['name'], state=state, noun=n, index=j))
                    img.append(z @ ai + offset_i + rng.standard_normal(dim_i))
                for t in range(4):
                    texts.append(dict(relation=r['name'], state=state, noun=n, template=t))
                    txt.append(z @ at + offset_t + 0.3 * rng.standard_normal(dim_t))
    img, txt = np.array(img), np.array(txt)
    return dict(relations=RELATIONS, nouns_for=nouns_for, images=images, texts=texts, img_feats=img,
                txt_feats=txt, img_q95=np.quantile(np.abs(img), .95, axis=1),
                txt_q95=np.quantile(np.abs(txt), .95, axis=1))


def test_assign_groups_cover_every_relation():
    data = synthetic()
    group = assign_groups(data['nouns_for'])
    for nouns in data['nouns_for'].values():
        assert {group[n] for n in nouns} == set(range(5))


def test_conditions_shapes_and_unit_norm():
    rng = np.random.default_rng(1)
    z = prepare(rng.standard_normal((200, 90)) + 3, 2.0)
    for cond in CONDITIONS:
        out = apply_condition(z, fit_condition(z, cond))
        assert np.allclose(np.linalg.norm(out, axis=1), 1)
        assert out.shape[1] == (90 if cond == 'center_only' else 64)


def test_uncentered_keeps_offset_and_reference_removes_it():
    rng = np.random.default_rng(2)
    z = prepare(rng.standard_normal((300, 70)) + 5, 10.0)
    ref = apply_condition(z, fit_condition(z, 'reference'))
    unc = apply_condition(z, fit_condition(z, 'uncentered_pca64'))
    assert diagnostics(ref)['mean_cosine'] < diagnostics(unc)['mean_cosine']


def test_whiten_gives_identity_covariance_before_renormalization():
    rng = np.random.default_rng(3)
    z = prepare(rng.standard_normal((300, 70)) * np.linspace(1, 5, 70), 10.0)
    fit = fit_condition(z, 'whiten_pca64')
    w = ((z - fit['mean']) @ fit['basis']) / fit['component_scale']
    assert np.allclose(np.cov(w.T), np.eye(64), atol=1e-6)


def test_rank_metrics_hand_example():
    sims = np.array([[0.9, 0.8, 0.1, 0.0]])
    is_target = np.array([False, True, True, False])
    cand_noun = np.array([0, 0, 0, 1])
    m = rank_metrics(sims, is_target, cand_noun, np.array([0]))
    assert m['state'][0] == 0 and m['noun'][0] == 1 and m['joint'][0] == 0
    assert m['ap'][0] == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_bootstrap_and_macro():
    a = {f'{c}|n{k}': 1.0 for c in ['cleaning', 'cooking'] for k in range(6)}
    b = {key: 0.5 for key in a}
    names = ['cleaning', 'cooking']
    assert macro(a, names) == 1.0
    est, low, high = bootstrap(a, b, names, draws=200)
    assert est == pytest.approx(0.5) and low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_analyze_end_to_end_on_synthetic_signal():
    data = synthetic()
    analysis = analyze(data, random_controls=5)
    names = [r['name'] for r in RELATIONS]
    ref = analysis['results']['reference']
    assert set(ref) == set(METHODS)
    ap = {m: macro(ref[m]['ap'], names) for m in METHODS}
    assert all(np.isfinite(v) for v in ap.values())
    assert ap['shared_text'] > ap['no_change'] > ap['reversed']
    for cond in CONDITIONS:
        assert len(analysis['diagnostics'][cond]) == 5
        assert macro(analysis['results'][cond]['shared_text']['ap'], names) > 0


def make_zip(path, folders):
    with zipfile.ZipFile(path, 'w') as zf:
        for folder, count in folders.items():
            for k in range(count):
                zf.writestr(f'release_dataset/images/{folder}/{k:04d}.jpg', f'{folder}-{k}'.encode())
        zf.writestr('__MACOSX/images/raw cat/junk.jpg', b'junk')


def test_scan_and_build_manifest_respect_minimum_and_duplicates(tmp_path):
    folders = {'raw chicken': 12, 'cooked chicken': 12, 'raw fish': 12, 'cooked fish': 9, 'dirty car': 12,
               'clean car': 12, 'sunny day': 12}
    archive = tmp_path / 'states.zip'
    make_zip(archive, folders)
    with zipfile.ZipFile(archive) as zf:
        found = scan_archive(zf.namelist())
    assert ('raw', 'chicken') in found and ('sunny', 'day') not in found and ('raw', 'cat') not in found
    manifest = build_manifest(archive)
    assert manifest['nouns_for']['cooking'] == ['chicken']   # fish has only 9 cooked images
    assert manifest['nouns_for']['cleaning'] == ['car']
    chicken = [r for r in manifest['images'] if r['noun'] == 'chicken' and r['state'] == 'raw']
    assert [r['index'] for r in chicken] == list(range(8))
    assert len({r['sha256'] for r in chicken}) == 8
    assert len([t for t in manifest['texts'] if t['noun'] == 'car']) == 2 * len(TEMPLATES)
    assert build_manifest(archive)['hash'] == manifest['hash']


def test_duplicate_images_are_skipped(tmp_path):
    archive = tmp_path / 'dups.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        for state in ('raw', 'cooked'):
            for k in range(14):
                zf.writestr(f'images/{state} egg/{k}.jpg', b'same' if k < 4 else f'{state}{k}'.encode())
    manifest = build_manifest(archive)
    raw = [r for r in manifest['images'] if r['state'] == 'raw']
    assert len({r['sha256'] for r in raw}) == 8


def test_load_modality_checks_rows(tmp_path):
    folder = tmp_path / 'features_vision'
    save_chunk(folder, 0, np.ones((3, 4)), np.ones(3))
    save_chunk(folder, 3, np.ones((2, 4)), np.ones(2))
    (folder / 'metadata.json').write_text('{"complete": true}')
    feats, q95 = load_modality(tmp_path, 'vision', 5)
    assert feats.shape == (5, 4) and q95.shape == (5,)
    with pytest.raises(ValueError):
        load_modality(tmp_path, 'vision', 6)