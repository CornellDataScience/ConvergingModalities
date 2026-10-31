"""Alignment of text-derived state directions with held-out visual state changes.

See STATE_DIRECTIONS_PROTOCOL.md.

    python state_directions.py [--run data/states] [--shuffles 200]
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from state_transfer import (CONDITIONS, GROUPS, apply_condition, assign_groups, fit_condition, fit_ridge,
                            load_run, prepare, unit)
from states_data import PER_STATE, SEED, write_json


def match_stats(m):
    """margin = mean diagonal - mean off-diagonal; top1 = fraction of rows whose largest entry is the diagonal."""
    k = len(m)
    diag = np.diag(m)
    off = (m.sum() - diag.sum()) / (m.size - k)
    return dict(margin=float(diag.mean() - off), top1=float((m.argmax(1) == np.arange(k)).mean()))


def cosine_matrix(rows, cols):
    return unit(rows) @ unit(cols).T


def build_index(data):
    img, txt = {}, {}
    for k, r in enumerate(data['images']):
        img.setdefault((r['state'], r['noun']), {})[r['index']] = k
    for k, r in enumerate(data['texts']):
        txt.setdefault((r['state'], r['noun']), {})[r['template']] = k
    return ({k: np.array([v[i] for i in sorted(v)]) for k, v in img.items()},
            {k: np.array([v[i] for i in sorted(v)]) for k, v in txt.items()})


def state_diff(space, index, spec, noun, first=0):
    src, tgt = index[(spec['source'], noun)][first:], index[(spec['target'], noun)][first:]
    return space[tgt].mean(0) - space[src].mean(0)


def bridge_inputs(text, image, txt_index, img_index, specs, train_nouns):
    xs, ys = [], []
    for c, spec in specs.items():
        for n in train_nouns[c]:
            for state in (spec['source'], spec['target']):
                rows = txt_index[(state, n)]
                xs.append(text[rows])
                ys.append(np.tile(image[img_index[(state, n)][:PER_STATE // 2]].mean(0), (len(rows), 1)))
    return np.vstack(xs), np.vstack(ys)


def text_directions(text, txt_index, specs, nouns, w):
    return np.stack([np.mean([state_diff(text, txt_index, spec, n) for n in nouns[c]], axis=0) @ w
                     for c, spec in specs.items()])


def run_fold(data, fold, cond, group, index, shuffles, splits, seed):
    img_index, txt_index = index
    names = [r['name'] for r in data['relations']]
    specs = {r['name']: r for r in data['relations']}
    nouns_for = data['nouns_for']
    test_g, val_g = fold, (fold + 1) % GROUPS
    img_group = np.array([group[r['noun']] for r in data['images']])
    txt_group = np.array([group[r['noun']] for r in data['texts']])
    train_img = np.where((img_group != test_g) & (img_group != val_g))[0]
    train_txt = np.where((txt_group != test_g) & (txt_group != val_g))[0]
    pi = prepare(data['img_feats'], data['img_q95'][train_img].mean())
    pt = prepare(data['txt_feats'], data['txt_q95'][train_txt].mean())
    image = apply_condition(pi, fit_condition(pi[train_img], cond))
    text = apply_condition(pt, fit_condition(pt[train_txt], cond))
    train = {c: [n for n in nouns_for[c] if group[n] not in (test_g, val_g)] for c in names}
    test = {c: [n for n in nouns_for[c] if group[n] == test_g] for c in names}
    x, y = bridge_inputs(text, image, txt_index, img_index, specs, train)
    w, _ = fit_ridge(x, y)
    d_text = text_directions(text, txt_index, specs, train, w)
    d_img = np.stack([np.mean([state_diff(image, img_index, specs[c], n, PER_STATE // 2) for n in train[c]], axis=0)
                      for c in names])
    d_test = np.stack([np.mean([state_diff(image, img_index, specs[c], n) for n in test[c]], axis=0) for c in names])
    m_text, m_img = cosine_matrix(d_text, d_test), cosine_matrix(d_img, d_test)
    rng = np.random.default_rng([seed, fold, CONDITIONS.index(cond)])
    null = []
    for _ in range(shuffles):
        ws, _ = fit_ridge(x, y[rng.permutation(len(y))])
        null.append(match_stats(cosine_matrix(text_directions(text, txt_index, specs, train, ws), d_test)))
    stab_t, stab_i = [], []
    for _ in range(splits):
        a_t, b_t, a_i, b_i = [], [], [], []
        for c in names:
            order = rng.permutation(len(train[c]))
            half = len(order) // 2
            parts = [[train[c][k] for k in order[:half]], [train[c][k] for k in order[half:]]]
            a_t.append(np.mean([state_diff(text, txt_index, specs[c], n) for n in parts[0]], axis=0) @ w)
            b_t.append(np.mean([state_diff(text, txt_index, specs[c], n) for n in parts[1]], axis=0) @ w)
            a_i.append(np.mean([state_diff(image, img_index, specs[c], n, PER_STATE // 2) for n in parts[0]], axis=0))
            b_i.append(np.mean([state_diff(image, img_index, specs[c], n, PER_STATE // 2) for n in parts[1]], axis=0))
        stab_t.append(float(np.mean(np.sum(unit(np.stack(a_t)) * unit(np.stack(b_t)), axis=1))))
        stab_i.append(float(np.mean(np.sum(unit(np.stack(a_i)) * unit(np.stack(b_i)), axis=1))))
    return dict(fold=fold, text=match_stats(m_text), image=match_stats(m_img), matrix_text=m_text.tolist(),
                matrix_image=m_img.tolist(), null_margin=[v['margin'] for v in null],
                null_top1=[v['top1'] for v in null], stability_text=float(np.mean(stab_t)),
                stability_image=float(np.mean(stab_i)))


def analyze(data, conditions=CONDITIONS, shuffles=200, splits=20, seed=SEED):
    group = assign_groups(data['nouns_for'], seed)
    index = build_index(data)
    return {c: [run_fold(data, f, c, group, index, shuffles, splits, seed) for f in range(GROUPS)]
            for c in conditions}


def summarize(folds):
    t_margin = [f['text']['margin'] for f in folds]
    null = np.mean([f['null_margin'] for f in folds], axis=0)
    null_top1 = np.mean([f['null_top1'] for f in folds], axis=0)
    obs, obs_top1 = float(np.mean(t_margin)), float(np.mean([f['text']['top1'] for f in folds]))
    return dict(text_margin=obs, text_margin_range=[min(t_margin), max(t_margin)], text_top1=obs_top1,
                null_margin_mean=float(null.mean()), null_margin_q95=float(np.quantile(null, .95)),
                margin_p=float((1 + (null >= obs).sum()) / (1 + len(null))),
                null_top1_mean=float(null_top1.mean()),
                image_margin=float(np.mean([f['image']['margin'] for f in folds])),
                image_top1=float(np.mean([f['image']['top1'] for f in folds])),
                stability_text=float(np.mean([f['stability_text'] for f in folds])),
                stability_image=float(np.mean([f['stability_image'] for f in folds])),
                matrix_text=np.mean([f['matrix_text'] for f in folds], axis=0).tolist())


def report(summary, names, out, shuffles):
    lines = ['# State directions: text-derived change vs held-out visual change', '',
             'Cosine between the bridged text direction (training categories) and the visual difference between states in held-out categories. Margin = mean matching-relation cosine minus mean other-relation cosine; top-1 chance = 0.20. Fold-averaged; ranges are over 5 dependent folds. Shuffled-bridge null: ' + f'{shuffles} shuffles.', '',
             '| Condition | Text margin [fold range] | Text top-1 | Shuffled-bridge margin mean / 95th pct | p | Image-derived margin | Image top-1 | Stability text / image |',
             '|---|---|---|---|---|---|---|---|']
    for cond, s in summary.items():
        lines.append(f"| {cond} | {s['text_margin']:.3f} [{s['text_margin_range'][0]:.3f}, {s['text_margin_range'][1]:.3f}] "
                     f"| {s['text_top1']:.2f} | {s['null_margin_mean']:.3f} / {s['null_margin_q95']:.3f} | {s['margin_p']:.3f} "
                     f"| {s['image_margin']:.3f} | {s['image_top1']:.2f} | {s['stability_text']:.2f} / {s['stability_image']:.2f} |")
    ref = 'reference' if 'reference' in summary else next(iter(summary))
    lines += ['', f'## Alignment matrix, {ref} (rows: text direction; columns: held-out visual direction)', '',
              '| | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for name, row in zip(names, summary[ref]['matrix_text']):
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in row) + ' |')
    lines += ['', 'Exploratory. Protocol: STATE_DIRECTIONS_PROTOCOL.md. Groups of images, not before-and-after pairs.']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot(summary, names, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = 'reference' if 'reference' in summary else next(iter(summary))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    shown = ax.imshow(np.array(summary[ref]['matrix_text']), cmap='viridis')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('held-out visual direction'); ax.set_ylabel('text direction')
    fig.colorbar(shown, ax=ax, label='cosine')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=Path('data/states'))
    parser.add_argument('--out', type=Path, default=Path('results/state_directions'))
    parser.add_argument('--shuffles', type=int, default=200)
    parser.add_argument('--splits', type=int, default=20)
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS, choices=CONDITIONS)
    args = parser.parse_args(argv)
    data, manifest = load_run(args.run)
    names = [r['name'] for r in data['relations']]
    folds = analyze(data, args.conditions, args.shuffles, args.splits)
    summary = {c: summarize(f) for c, f in folds.items()}
    own = [Path(__file__), Path(__file__).with_name('STATE_DIRECTIONS_PROTOCOL.md')]
    write_json(args.out / 'results.json', dict(
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in own},
        manifest_hash=manifest.get('hash'), shuffles=args.shuffles, splits=args.splits,
        summary=summary, folds=folds))
    report(summary, names, args.out, args.shuffles)
    plot(summary, names, args.out / 'alignment_matrix.png')
    print((args.out / 'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
