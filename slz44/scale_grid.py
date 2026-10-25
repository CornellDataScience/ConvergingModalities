"""Metric x control x sample-size grid. See SCALE_GRID_PROTOCOL.md.

    python scale_grid.py [--sizes 128 256 512] [--permutations 200]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anisotropy import MODES, digest, measure, write_json
from extra_metrics import cka_with_null, gw_cost_with_null, normalized_distance
from metric_controls import prepare_vectors

METRICS = [('knn_fixed', 'k-NN excess, k=10'), ('knn_scaled', 'k-NN excess, k scaled with N'),
           ('cka', 'CKA excess'), ('gw', 'GW cost reduction')]


def scaled_k(n, pool, base=10):
    return max(1, round(base * (n - 1) / (pool - 1)))


def score_subset(a, b, n, pool, seed, permutations):
    t = lambda x: torch.from_numpy(x).float()
    fixed = measure(t(a), t(b), 10, seed + 10000, permutations)
    scaled = measure(t(a), t(b), scaled_k(n, pool), seed + 10000, permutations)
    cka = cka_with_null(a, b, permutations, seed + 20000)
    gw = gw_cost_with_null(normalized_distance(a), normalized_distance(b), permutations, seed + 30000)
    return dict(knn_fixed=fixed['excess'], knn_scaled=scaled['excess'], cka=cka['excess'], gw=gw['reduction'],
                knn_fixed_raw=fixed['score'], knn_fixed_null=fixed['null_mean'], k_scaled=scaled_k(n, pool))


def summarize(rows, sizes):
    def key(r): return (r['n'], r['seed'], r['language'], r['vision'])
    base = {key(r): r for r in rows if r['mode'] == 'baseline'}
    lines = ['# Metric x control x sample-size grid', '',
             'Nested random subsets of the 512 evaluation rows; frozen layers; 200 shuffled pairings. Values are mean [range] over 18 rows (dependent; not confidence intervals). "up" = rows of 18 above baseline.', '']
    for metric, label in METRICS:
        lines += [f'## {label}', '', '| Condition | ' + ' | '.join(f'N={n}' for n in sizes) + ' |', '|---|' + '---|' * len(sizes)]
        for mode in MODES:
            cells = []
            for n in sizes:
                rr = [r for r in rows if r['mode'] == mode and r['n'] == n]
                vals = [r[metric] for r in rr]
                up = '' if mode == 'baseline' else f"; up {sum(r[metric] > base[key(r)][metric] for r in rr)}/{len(rr)}"
                cells.append(f'{np.mean(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}]{up}')
            lines.append(f'| {mode} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    raw = [(n, np.mean([r['knn_fixed_raw'] for r in rows if r['mode'] == 'baseline' and r['n'] == n]),
            np.mean([r['knn_fixed_null'] for r in rows if r['mode'] == 'baseline' and r['n'] == n])) for n in sizes]
    lines += ['## Raw k-NN overlap vs null at fixed k=10 (baseline)', '', '| N | raw overlap | shuffled mean |', '|---|---|---|']
    lines += [f'| {n} | {a:.4f} | {b:.4f} |' for n, a, b in raw]
    lines += ['', 'Exploratory. Protocol: SCALE_GRID_PROTOCOL.md. Per-row values: scores.csv.']
    return '\n'.join(lines) + '\n'


def plot(rows, sizes, path):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6), sharex=True)
    for ax, (metric, label) in zip(axes, METRICS):
        for mode in MODES:
            ax.plot(sizes, [np.mean([r[metric] for r in rows if r['mode'] == mode and r['n'] == n]) for n in sizes],
                    marker='o', label=mode)
        ax.set_xscale('log', base=2); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
        ax.set_title(label, fontsize=9); ax.set_xlabel('evaluation pool size N')
    axes[0].set_ylabel('mean over 18 rows')
    axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, default=Path('data/baseline'))
    parser.add_argument('--heldout', type=Path, default=Path('data/heldout_layers/results.json'))
    parser.add_argument('--out', type=Path, default=Path('results/scale_grid'))
    parser.add_argument('--sizes', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--permutations', type=int, default=200)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    previous = json.loads(args.heldout.read_text())
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    assert digest(manifest) == previous['manifest_hash']
    vectors = prepare_vectors(args.baseline, previous, manifest)
    rows = []
    for old in previous['results']:
        seed = old['seed']
        pool = len(vectors[old['language'], seed, old['layers'][0], 'baseline'])
        order = np.random.default_rng(seed + 777).permutation(pool)
        assert max(args.sizes) <= pool
        for n in args.sizes:
            subset = order[:n]
            for mode in MODES:
                a, b = [vectors[old[name], seed, old['layers'][i], mode][subset] for i, name in enumerate(['language', 'vision'])]
                rows.append(dict(mode=mode, n=n, seed=seed, language=old['language'], vision=old['vision'],
                                 **score_subset(a, b, n, pool, seed, args.permutations)))
        print('Scored', seed, old['language'], '/', old['vision'], flush=True)
    own = [Path(__file__), Path(__file__).with_name('SCALE_GRID_PROTOCOL.md'), args.heldout]
    write_json(args.out / 'results.json', dict(
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in own},
        manifest_hash=digest(manifest), sizes=args.sizes, permutations=args.permutations, rows=rows))
    with (args.out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = summarize(rows, args.sizes)
    (args.out / 'REPORT.md').write_text(report, encoding='utf-8')
    plot(rows, args.sizes, args.out / 'scale_grid.png')
    print(report)


if __name__ == '__main__':
    main()
