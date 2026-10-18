"""Anisotropy-control sensitivity study; see ANISOTROPY_PROTOCOL.md.

Self-contained. Reads the cached features and held-out layer choices produced by
prepare_data.py (--baseline, --heldout); its baseline check asserts that the
scores and null draws recomputed here match the held-out results file.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

MODES = ['baseline', 'center', 'pca128', 'standardize', 'remove3', 'pca128_whiten']
DIAGNOSTICS = ['mean_cosine', 'effective_rank', 'top_dim_share', 'top_pc_share']


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def load_features(run, spec, manifest):
    """Cached [samples, layers, dimensions] features written by the baseline extraction."""
    folder = run / 'features' / spec['id'].replace('/', '__')
    metadata = json.loads((folder / 'metadata.json').read_text())
    if not metadata['complete'] or metadata['identity']['sample_hash'] != manifest['sample_hash']:
        raise ValueError('Incomplete or mismatched feature cache')
    arrays, rows = [], []
    for file in sorted(folder.glob('*.npz')):
        with np.load(file) as chunk:
            arrays.append(chunk['feats'])
            rows.extend(chunk['rows'].tolist())
    if rows != list(range(manifest['samples'])):
        raise ValueError('Missing, duplicated, or reordered sample rows')
    return torch.from_numpy(np.concatenate(arrays))


def neighbors(x, k):
    if x.ndim != 2 or not 0 < k < len(x):
        raise ValueError('Need [N,D] features and 0 < k < N')
    similarities = x @ x.T
    similarities.fill_diagonal_(-float('inf'))
    return similarities.argsort(dim=1, descending=True)[:, :k].numpy()


def overlap(a, b):
    if a.shape != b.shape:
        raise ValueError('Neighbor arrays must have equal shapes')
    return (a[:, :, None] == b[:, None, :]).any(axis=2).mean().item()


def measure(a, b, k, seed, permutations=1000):
    left, right = neighbors(a, k), neighbors(b, k)
    score = overlap(left, right)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        p = rng.permutation(len(right))
        null.append(overlap(left, np.argsort(p)[right[p]]))
    mu = float(np.mean(null))
    return dict(score=score, null_mean=mu, null_q95=float(np.quantile(null, .95)),
                excess=score - mu, adjusted=(score - mu) / (1 - mu),
                p=(1 + sum(v >= score for v in null)) / (1 + permutations), null=null)


def fit_control(train, mode, dimensions=128, top=3):
    """Fit on selection rows only. train: clipped, L2-normalized [N, D]."""
    if mode == 'baseline':
        return dict(mode=mode, retained=None)
    mean = train.mean(0)
    centered = train - mean
    if mode == 'center':
        return dict(mode=mode, mean=mean, retained=None)
    if mode == 'standardize':
        return dict(mode=mode, mean=mean, scale=centered.std(0).clamp_min(1e-8), retained=None)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular.square()
    if mode == 'pca128':
        return dict(mode=mode, mean=mean, basis=vh[:dimensions].T.contiguous(),
                    retained=float(energy[:dimensions].sum() / energy.sum()))
    if mode == 'remove3':
        return dict(mode=mode, mean=mean, basis=vh[:top].T.contiguous(),
                    retained=float(1 - energy[:top].sum() / energy.sum()))
    if mode == 'pca128_whiten':
        scale = (singular[:dimensions] / np.sqrt(len(train) - 1)).clamp_min(1e-8)
        return dict(mode=mode, mean=mean, basis=vh[:dimensions].T.contiguous(), scale=scale,
                    retained=float(energy[:dimensions].sum() / energy.sum()))
    raise ValueError(mode)


def apply_control(x, fitted):
    mode = fitted['mode']
    if mode == 'baseline':
        return x
    x = x - fitted['mean']
    if mode == 'standardize':
        x = x / fitted['scale']
    elif mode == 'remove3':
        x = x - (x @ fitted['basis']) @ fitted['basis'].T
    elif mode == 'pca128':
        x = x @ fitted['basis']
    elif mode == 'pca128_whiten':
        x = (x @ fitted['basis']) / fitted['scale']
    return F.normalize(x, dim=-1)


def diagnostics(x):
    """Geometry of L2-normalized evaluation vectors [N, D]."""
    n = len(x)
    similarities = x @ x.T
    singular = torch.linalg.svdvals(x - x.mean(0))
    p = singular / singular.sum()
    variance = x.var(0)
    return dict(mean_cosine=float((similarities.sum() - similarities.diag().sum()) / (n * (n - 1))),
                effective_rank=float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())),
                top_dim_share=float(variance.max() / variance.sum()),
                top_pc_share=float(singular[0].square() / singular.square().sum()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('data/baseline'),
                        help='run directory with manifest.json and features/ (see prepare_data.py)')
    parser.add_argument('--heldout', type=Path, default=Path('data/heldout_layers/results.json'),
                        help='held-out layer results: splits, thresholds, selected layers (see prepare_data.py)')
    parser.add_argument('--out', type=Path, default=Path('results/anisotropy'))
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    previous = json.loads(args.heldout.read_text())
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    assert digest(manifest) == previous['manifest_hash']
    vectors = {}
    for spec in manifest['config']['models']:
        name = spec['id']
        raw = load_features(args.baseline, spec, manifest)
        for s, split in enumerate(previous['splits']):
            layers = set()
            for r in previous['results']:
                if r['seed'] == split['seed']:
                    if r['language'] == name: layers.add(r['layers'][0])
                    if r['vision'] == name: layers.add(r['layers'][1])
            threshold = previous['thresholds'][name][s]
            for layer in layers:
                train = F.normalize(raw[split['selection'], layer].clamp(-threshold, threshold), dim=-1)
                evaluation = F.normalize(raw[split['evaluation'], layer].clamp(-threshold, threshold), dim=-1)
                for mode in MODES:
                    fitted = fit_control(train, mode)
                    x = apply_control(evaluation, fitted)
                    vectors[name, split['seed'], layer, mode] = (x, fitted['retained'], diagnostics(x))
        del raw
        print('Prepared ' + name, flush=True)
    results = []
    for old in previous['results']:
        seed = old['seed']
        for mode in MODES:
            (a, ra, da), (b, rb, db) = [vectors[old[name], seed, old['layers'][i], mode]
                                        for i, name in enumerate(['language', 'vision'])]
            measured = measure(a, b, 10, seed + 10000)
            if mode == 'baseline':
                # Guards the reimplemented helpers against the original held-out run.
                assert np.isclose(measured['score'], old['evaluation_score'], atol=1e-12)
                np.testing.assert_allclose(measured['null'], old['null'], atol=1e-12)
            row = dict(study='anisotropy', mode=mode, n=512, k=10, fraction=10 / 511, seed=seed,
                       language=old['language'], vision=old['vision'], layers=old['layers'],
                       retained_language=ra, retained_vision=rb, **measured)
            for key in DIAGNOSTICS:
                row[key + '_language'] = da[key]
                row[key + '_vision'] = db[key]
            results.append(row)
        print('Scored ' + str(seed) + ' ' + old['language'] + ' / ' + old['vision'], flush=True)
    own = [Path(__file__), Path(__file__).with_name('ANISOTROPY_PROTOCOL.md'), args.heldout]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in own}
    write_json(args.out / 'results.json', dict(source_hashes=hashes, manifest_hash=digest(manifest),
        splits=previous['splits'], permutations=1000, modes=MODES, results=results))
    fields = ['study', 'mode', 'seed', 'language', 'vision', 'layers', 'n', 'k', 'fraction', 'score',
              'null_mean', 'null_q95', 'excess', 'adjusted', 'p', 'retained_language', 'retained_vision']
    fields += [key + side for key in DIAGNOSTICS for side in ('_language', '_vision')]
    with (args.out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    summarize(results, args.out)


def matched(results, row, mode):
    return next(r for r in results if r['mode'] == mode and all(r[x] == row[x] for x in ['seed', 'language', 'vision']))


def summarize(results, out):
    lines = ['# Anisotropy-control sensitivity', '',
        'Three overlapping splits; six model pairs; N=512, k=10. Layers and clipping fixed from selection halves; every control fitted on selection rows only. Each condition uses the same 1,000 matched permutations. Ranges are across dependent comparisons, not confidence intervals.', '',
        '| Condition | Raw range | Excess range | Mean change vs baseline | Increases / 18 | Strict size increases / 6 | Mean cosine (lang / vision) | Effective rank (lang / vision) | Top-PC share (lang / vision) |',
        '|---|---|---|---:|---:|---:|---|---|---|']
    for mode in MODES:
        rr = [r for r in results if r['mode'] == mode]
        changes = [r['score'] - matched(results, r, 'baseline')['score'] for r in rr]
        monotonic = 0
        for seed in sorted(set(r['seed'] for r in rr)):
            for size in ['small', 'base']:
                ordered = [next(r for r in rr if r['seed'] == seed and size in r['vision'] and r['language'].endswith(lang)) for lang in ['560m', '1b1', '1b7']]
                monotonic += all(a['score'] < b['score'] for a, b in zip(ordered, ordered[1:]))
        ranges = ['%.4f–%.4f' % (min(r[field] for r in rr), max(r[field] for r in rr)) for field in ['score', 'excess']]
        pairs = ['%.3f / %.3f' % (np.mean([r[key + '_language'] for r in rr]), np.mean([r[key + '_vision'] for r in rr]))
                 for key in ['mean_cosine', 'effective_rank', 'top_pc_share']]
        lines.append(f'| {mode} | ' + ' | '.join(ranges) + f' | {np.mean(changes):.6f} | {sum(x > 0 for x in changes)} | {monotonic} | ' + ' | '.join(pairs) + ' |')
    lines += ['', '## Matched differences', '']
    for label, high, low in [('standardize minus baseline', 'standardize', 'baseline'),
                             ('remove3 minus baseline', 'remove3', 'baseline'),
                             ('pca128_whiten minus baseline', 'pca128_whiten', 'baseline'),
                             ('pca128_whiten minus pca128 (whitening alone)', 'pca128_whiten', 'pca128'),
                             ('remove3 minus center (top-direction removal alone)', 'remove3', 'center')]:
        changes = [r['score'] - matched(results, r, low)['score'] for r in results if r['mode'] == high]
        lines.append(f'- {label}: mean {np.mean(changes):.6f}; range {min(changes):.6f} to {max(changes):.6f}; increases in {sum(x > 0 for x in changes)}/18 comparisons.')
    lines += ['', f"Stored condition scores above their corresponding null 95th percentile: {sum(r['score'] > r['null_q95'] for r in results)}/{len(results)}.", '',
        'Diagnostics are means over the 18 rows, where each model appears in several rows; they describe geometry and do not show that anisotropy caused a score change. Baseline scores and all null draws match the earlier held-out run. Layers are baseline-selected, not each control\'s best. PCA and whitening are fitted per modality; whitening is inside the 128-dimensional subspace because the 512-row selection covariance is rank deficient. Per-dimension scales from 512 rows are noisy. No claims about N>512, independent replications, or statistically established differences.', '',
        'Protocol: ANISOTROPY_PROTOCOL.md. Reproduce: `python anisotropy.py --baseline <run> --heldout <results.json>`. Full draws/provenance: results.json; detailed rows: scores.csv.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    positions = np.arange(len(MODES))
    for size, color, shift in [('small', 'tab:blue', -.15), ('base', 'tab:orange', .15)]:
        values = [[r['excess'] for r in results if r['mode'] == mode and size in r['vision']] for mode in MODES]
        means = np.array([np.mean(v) for v in values])
        axes[0].errorbar(positions + shift, means, yerr=[means - [min(v) for v in values], [max(v) for v in values] - means],
                         fmt='o', capsize=3, color=color, label='DINOv2 ' + size)
    axes[0].set_xticks(positions, MODES, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Overlap minus shuffled mean'); axes[0].legend(fontsize=8)
    axes[0].set_title('Alignment by control (means and ranges, not CIs)')
    for side, marker in [('language', 's'), ('vision', 'o')]:
        axes[1].plot(positions, [np.mean([r['mean_cosine_' + side] for r in results if r['mode'] == mode]) for mode in MODES], marker=marker, label=side)
    axes[1].set_xticks(positions, MODES, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Mean pairwise cosine'); axes[1].legend(fontsize=8)
    axes[1].set_title('Anisotropy diagnostic after control')
    fig.tight_layout(); fig.savefig(out / 'anisotropy.png', dpi=160); plt.close(fig)


if __name__ == '__main__': main()