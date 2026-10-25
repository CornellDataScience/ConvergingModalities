"""Do anisotropy controls change linear CKA, Gromov-Wasserstein structure cost, and blind matching?

See METRIC_CONTROLS_PROTOCOL.md. Reads the same cached inputs as anisotropy.py and scores the same
18 (split x model pair) rows under the same six conditions, with three metrics beyond mutual k-NN.

    python metric_controls.py [--gw-match]      # --gw-match needs POT (pip install pot)
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from anisotropy import MODES, apply_control, digest, fit_control, load_features, measure, write_json
from extra_metrics import cka_with_null, gw_cost_with_null, gw_match, normalized_distance

METRICS = ['knn', 'cka', 'gw']


def prepare_vectors(baseline, previous, manifest):
    vectors = {}
    for spec in manifest['config']['models']:
        name = spec['id']
        raw = load_features(baseline, spec, manifest)
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
                    vectors[name, split['seed'], layer, mode] = apply_control(evaluation, fit_control(train, mode)).double().numpy()
        del raw
        print('Prepared ' + name, flush=True)
    return vectors


def score_row(a, b, seed, permutations, do_match):
    row = {}
    knn = measure(torch.from_numpy(a).float(), torch.from_numpy(b).float(), 10, seed + 10000, permutations)
    cka = cka_with_null(a, b, permutations, seed + 20000)
    c1, c2 = normalized_distance(a), normalized_distance(b)
    gw = gw_cost_with_null(c1, c2, permutations, seed + 30000)
    row.update(knn_score=knn['score'], knn_excess=knn['excess'], cka_score=cka['score'], cka_excess=cka['excess'],
               gw_cost=gw['cost'], gw_null_mean=gw['null_mean'], gw_reduction=gw['reduction'])
    if do_match:
        match = gw_match(c1, c2)
        row.update(match_top1=match['top1'], match_top5=match['top5'])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, default=Path('data/baseline'))
    parser.add_argument('--heldout', type=Path, default=Path('data/heldout_layers/results.json'))
    parser.add_argument('--out', type=Path, default=Path('results/metric_controls'))
    parser.add_argument('--permutations', type=int, default=200)
    parser.add_argument('--gw-match', action='store_true', help='also run GW blind matching (needs POT)')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    previous = json.loads(args.heldout.read_text())
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    assert digest(manifest) == previous['manifest_hash']
    vectors = prepare_vectors(args.baseline, previous, manifest)
    rows = []
    for old in previous['results']:
        seed = old['seed']
        for mode in MODES:
            a, b = [vectors[old[name], seed, old['layers'][i], mode] for i, name in enumerate(['language', 'vision'])]
            row = dict(mode=mode, seed=seed, language=old['language'], vision=old['vision'],
                       **score_row(a, b, seed, args.permutations, args.gw_match))
            rows.append(row)
        print('Scored', seed, old['language'], '/', old['vision'], flush=True)
    own = [Path(__file__), Path(__file__).with_name('METRIC_CONTROLS_PROTOCOL.md'), args.heldout]
    write_json(args.out / 'results.json', dict(
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in own},
        manifest_hash=digest(manifest), permutations=args.permutations, rows=rows))
    with (args.out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report(rows, args.out, args.gw_match)
    print((args.out / 'REPORT.md').read_text(encoding='utf-8'))


def report(rows, out, matched):
    def match(row, mode):
        return next(r for r in rows if r['mode'] == mode and all(r[k] == row[k] for k in ['seed', 'language', 'vision']))
    cols = [('knn_excess', 'k-NN excess'), ('cka_excess', 'CKA excess'), ('gw_reduction', 'GW cost reduction')]
    if matched:
        cols += [('match_top1', 'GW blind-match top-1')]
    lines = ['# Anisotropy controls under CKA, Gromov-Wasserstein cost and blind matching', '',
             'Same 18 rows (3 splits x 6 model pairs), layers and fitted controls as the k-NN study. Excess = score minus shuffled-pairing mean; GW reduction = (null cost - observed cost) / null cost. Ranges are across dependent rows, not confidence intervals.', '',
             '| Condition | ' + ' | '.join(f'{label} mean [range]; up vs baseline /18' for _, label in cols) + ' |',
             '|---|' + '---|' * len(cols)]
    for mode in MODES:
        rr = [r for r in rows if r['mode'] == mode]
        cells = []
        for key, _ in cols:
            vals = [r[key] for r in rr]
            ups = sum(r[key] > match(r, 'baseline')[key] for r in rr) if mode != 'baseline' else '-'
            cells.append(f'{np.mean(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}]; {ups}')
        lines.append(f'| {mode} | ' + ' | '.join(cells) + ' |')
    if matched:
        lines += ['', f'Blind-matching chance top-1 is 1/512 = {1/512:.4f}.']
    lines += ['', '## Agreement between metrics', '']
    for (ka, la), (kb, lb) in [(cols[0], cols[1]), (cols[0], cols[2]), (cols[1], cols[2])]:
        deltas = lambda key: [r[key] - match(r, 'baseline')[key] for r in rows if r['mode'] != 'baseline']
        lines.append(f'- Change vs baseline, {la} vs {lb}: Pearson r = {np.corrcoef(deltas(ka), deltas(kb))[0, 1]:+.3f} over 90 condition-rows (descriptive; rows are dependent).')
    lines += ['', 'Exploratory. Protocol: METRIC_CONTROLS_PROTOCOL.md. Per-row values: scores.csv.']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
