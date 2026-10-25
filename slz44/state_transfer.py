"""Preprocessing sensitivity of text-to-image state transfer; see STATE_CONTROLS_PROTOCOL.md.

A text-derived relation direction (e.g. raw -> cooked) is mapped into image space with a ridge
bridge and added to held-out image queries. Only the preprocessing applied to both modalities
changes between conditions.

    python state_transfer.py --run data/states --out results/state_controls
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from states_data import PER_STATE, RELATIONS, SEED, load_modality, write_json

CONDITIONS = ['reference', 'uncentered_pca64', 'center_only', 'standardize_pca64',
              'remove3_pca64', 'whiten_pca64']
METHODS = ['no_change', 'shared_text', 'image_derived', 'reversed', 'random', 'wrong_change', 'direct_text']
METRICS = ['ap', 'joint', 'state', 'noun']
SCALES = [0, .25, .5, 1, 2, 4]
DIMENSIONS = 64
TOP = 3
ALPHA = 1.0
GROUPS = 5


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def prepare(x, threshold):
    """Clip at the training threshold, then L2-normalize."""
    return unit(np.clip(x, -threshold, threshold))


def fit_condition(z, condition, dimensions=DIMENSIONS, top=TOP):
    """Fit one preprocessing condition on prepared training rows z [N, D]."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    mean = z.mean(0)
    c = z - mean
    fit = dict(center=condition != 'uncentered_pca64', mean=mean, scale=None, remove=None,
               basis=None, component_scale=None)
    if condition == 'standardize_pca64':
        fit['scale'] = np.maximum(c.std(0), 1e-8)
        c = c / fit['scale']
    if condition == 'remove3_pca64':
        fit['remove'] = np.linalg.svd(c, full_matrices=False)[2][:top].T
        c = c - (c @ fit['remove']) @ fit['remove'].T
    if condition != 'center_only':
        _, s, vh = np.linalg.svd(c, full_matrices=False)
        fit['basis'] = vh[:dimensions].T
        if condition == 'whiten_pca64':
            fit['component_scale'] = np.maximum(s[:dimensions] / np.sqrt(len(z) - 1), 1e-8)
    return fit


def apply_condition(x, fit):
    if fit['center']:
        x = x - fit['mean']
    if fit['scale'] is not None:
        x = x / fit['scale']
    if fit['remove'] is not None:
        x = x - (x @ fit['remove']) @ fit['remove'].T
    if fit['basis'] is not None:
        x = x @ fit['basis']
    if fit['component_scale'] is not None:
        x = x / fit['component_scale']
    return unit(x)


def diagnostics(x):
    n = len(x)
    sims = x @ x.T
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    var = x.var(0)
    return dict(mean_cosine=float((sims.sum() - np.trace(sims)) / (n * (n - 1))),
                effective_rank=float(np.exp(-(p * np.log(p)).sum())),
                top_dim_share=float(var.max() / var.sum()),
                top_pc_share=float(s[0] ** 2 / (s ** 2).sum()))


def fit_ridge(x, y, alpha=ALPHA):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    if x.shape[1] <= x.shape[0]:
        w = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    else:
        w = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(x.shape[0]), yc)
    return w, ym - xm @ w


def assign_groups(nouns_for, seed=SEED, groups=GROUPS, attempts=10000):
    """Randomly split category labels into balanced groups; every relation needs each group."""
    nouns = sorted({n for v in nouns_for.values() for n in v})
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        order = rng.permutation(len(nouns))
        group = {nouns[k]: pos % groups for pos, k in enumerate(order)}
        if all({group[n] for n in v} == set(range(groups)) for v in nouns_for.values()):
            return group
    raise ValueError('No group assignment covers every relation; need more categories per relation')


def rank_metrics(sims, is_target, cand_noun, query_noun):
    relevant = is_target[None, :] & (cand_noun[None, :] == query_noun[:, None])
    order = np.argsort(-sims, axis=1, kind='stable')
    ranked = np.take_along_axis(relevant, order, axis=1)
    precision = np.cumsum(ranked, axis=1) / np.arange(1, sims.shape[1] + 1)
    top = order[:, 0]
    rows = np.arange(len(top))
    return dict(ap=(ranked * precision).sum(1) / relevant.sum(1),
                joint=relevant[rows, top].astype(float),
                state=is_target[top].astype(float),
                noun=(cand_noun[top] == query_noun).astype(float))


def analyze(data, conditions=CONDITIONS, random_controls=100, scales=SCALES, seed=SEED):
    relations, nouns_for = data['relations'], data['nouns_for']
    names = [r['name'] for r in relations]
    spec = {r['name']: r for r in relations}
    group = assign_groups(nouns_for, seed)
    noun_id = {n: i for i, n in enumerate(sorted(group))}
    img_index, txt_index = {}, {}
    for k, r in enumerate(data['images']):
        img_index.setdefault((r['state'], r['noun']), {})[r['index']] = k
    for k, r in enumerate(data['texts']):
        txt_index.setdefault((r['state'], r['noun']), {})[r['template']] = k
    img_index = {k: np.array([v[i] for i in sorted(v)]) for k, v in img_index.items()}
    txt_index = {k: np.array([v[i] for i in sorted(v)]) for k, v in txt_index.items()}
    img_group = np.array([group[r['noun']] for r in data['images']])
    txt_group = np.array([group[r['noun']] for r in data['texts']])
    per = PER_STATE
    results = {c: {m: {k: {} for k in METRICS} for m in METHODS} for c in conditions}
    diag = {c: [] for c in conditions}
    chosen = {c: [] for c in conditions}

    def pools(c, nouns):
        src, tgt = spec[c]['source'], spec[c]['target']
        out = []
        for i in range(per):
            cand, is_t, cn = [], [], []
            for m in nouns:
                for j in range(per):
                    if j != i:
                        cand.append(img_index[(src, m)][j]); is_t.append(False); cn.append(noun_id[m])
                    if j != (i + 1) % per:
                        cand.append(img_index[(tgt, m)][j]); is_t.append(True); cn.append(noun_id[m])
            out.append((np.array(cand), np.array(is_t), np.array(cn)))
        return out

    for fold in range(GROUPS):
        test_g, val_g = fold, (fold + 1) % GROUPS
        train_img = np.where((img_group != test_g) & (img_group != val_g))[0]
        train_txt = np.where((txt_group != test_g) & (txt_group != val_g))[0]
        pi = prepare(data['img_feats'], data['img_q95'][train_img].mean())
        pt = prepare(data['txt_feats'], data['txt_q95'][train_txt].mean())
        for ci, cond in enumerate(conditions):
            image = apply_condition(pi, fit_condition(pi[train_img], cond))
            text = apply_condition(pt, fit_condition(pt[train_txt], cond))
            train_c = {c: [n for n in nouns_for[c] if group[n] not in (test_g, val_g)] for c in names}
            xs, ys = [], []
            for c in names:
                for n in train_c[c]:
                    for state in (spec[c]['source'], spec[c]['target']):
                        rows = txt_index[(state, n)]
                        xs.append(text[rows])
                        ys.append(np.tile(image[img_index[(state, n)][:per // 2]].mean(0), (len(rows), 1)))
            w, b = fit_ridge(np.vstack(xs), np.vstack(ys))
            d_text, d_img = {}, {}
            for c in names:
                src, tgt = spec[c]['source'], spec[c]['target']
                d_text[c] = np.mean([text[txt_index[(tgt, n)]].mean(0) - text[txt_index[(src, n)]].mean(0)
                                     for n in train_c[c]], axis=0) @ w
                d_img[c] = np.mean([image[img_index[(tgt, n)][per // 2:]].mean(0)
                                    - image[img_index[(src, n)][per // 2:]].mean(0)
                                    for n in train_c[c]], axis=0)
            rng = np.random.default_rng([seed, fold, ci])
            rand = unit(rng.standard_normal((random_controls, image.shape[1])))

            def collect(c, nouns, pool, make):
                qn = np.array([noun_id[n] for n in nouns])
                total = {k: 0. for k in METRICS}
                for i, (cand, is_t, cn) in enumerate(pool):
                    y = np.stack([image[img_index[(spec[c]['source'], n)][i]] for n in nouns])
                    q = make(y)
                    k_, nq, d = q.shape
                    m = rank_metrics(q.reshape(-1, d) @ image[cand].T, is_t, cn, np.tile(qn, k_))
                    for k in METRICS:
                        total[k] = total[k] + m[k].reshape(k_, nq).mean(0)
                return {k: v / len(pool) for k, v in total.items()}

            def pick(c, vals, vpool, direction):
                best, best_ap = 0, -1.
                for s in scales:
                    ap = collect(c, vals, vpool, lambda y: unit(y[None] + s * direction))['ap'].mean()
                    if ap > best_ap + 1e-12:
                        best, best_ap = s, ap
                return best

            fold_scales = {}
            for c in names:
                src, tgt = spec[c]['source'], spec[c]['target']
                vals = [n for n in nouns_for[c] if group[n] == val_g]
                tests = [n for n in nouns_for[c] if group[n] == test_g]
                s_t = pick(c, vals, pools(c, vals), d_text[c])
                s_i = pick(c, vals, pools(c, vals), d_img[c])
                fold_scales[c] = dict(shared_text=s_t, image_derived=s_i)
                norm = np.linalg.norm(d_text[c])
                wrong = np.stack([unit(d_text[o]) * norm for o in names if o != c])[:, None, :]
                direct = unit(np.stack([text[txt_index[(tgt, n)]].mean(0) for n in tests]) @ w + b)
                makers = {'no_change': lambda y: y[None],
                          'shared_text': lambda y: unit(y[None] + s_t * d_text[c]),
                          'image_derived': lambda y: unit(y[None] + s_i * d_img[c]),
                          'reversed': lambda y: unit(y[None] - s_t * d_text[c]),
                          'random': lambda y: unit(y[None] + s_t * norm * rand[:, None, :]),
                          'wrong_change': lambda y: unit(y[None] + s_t * wrong),
                          'direct_text': lambda y: direct[None]}
                tpool = pools(c, tests)
                for method, make in makers.items():
                    out = collect(c, tests, tpool, make)
                    for k in METRICS:
                        for n, v in zip(tests, out[k]):
                            results[cond][method][k][f'{c}|{n}'] = float(v)
            chosen[cond].append(fold_scales)
            ti = np.where(img_group == test_g)[0]
            tt = np.where(txt_group == test_g)[0]
            diag[cond].append(dict(fold=fold, image=diagnostics(image[ti]), text=diagnostics(text[tt])))
    return dict(results=results, diagnostics=diag, scales=chosen)


def macro(values, names):
    """Mean within category, then within relation, then equally across relations."""
    return float(np.mean([np.mean([v for k, v in values.items() if k.split('|')[0] == c]) for c in names]))


def bootstrap(a, b, names, draws=2000, seed=0):
    """Category-label bootstrap of macro(a) - macro(b), conditional on per-category means."""
    keys = sorted(a)
    nouns = sorted({k.split('|')[1] for k in keys})
    col = {n: i for i, n in enumerate(nouns)}
    m = np.full((len(names), len(nouns)), np.nan)
    for k in keys:
        c, n = k.split('|')
        m[names.index(c), col[n]] = a[k] - b[k]
    support = ~np.isnan(m)
    counts = np.random.default_rng(seed).multinomial(len(nouns), np.full(len(nouns), 1 / len(nouns)), size=draws)
    num, den = counts @ np.where(support, m, 0).T, counts @ support.T
    per = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    low, high = np.percentile(np.nanmean(per, axis=1), [2.5, 97.5])
    return float(np.nanmean(np.nanmean(m, axis=1))), float(low), float(high)


def load_run(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    img, img_q = load_modality(run, 'vision', len(manifest['images']))
    txt, txt_q = load_modality(run, 'language', len(manifest['texts']))
    return dict(relations=manifest['config']['relations'], nouns_for=manifest['nouns_for'],
                images=manifest['images'], texts=manifest['texts'], img_feats=img, img_q95=img_q,
                txt_feats=txt, txt_q95=txt_q), manifest


def report(analysis, names, out, conditions):
    res = analysis['results']
    rows, lines = [], ['# State-transfer preprocessing sensitivity', '',
        'Own sample of MIT-States, five relations, five category-held-out folds. Only the preprocessing differs between conditions. Intervals are descriptive category-label bootstrap intervals (2,000 draws, conditional on per-category means), not adjusted for the six conditions.', '',
        '| Condition | AP no change | AP shared | AP gain [95%] | Category kept, no change | Category kept, shared | Change [95%] | Joint no change | Joint shared | Mean cosine (image / text) |',
        '|---|---:|---:|---|---:|---:|---|---:|---:|---|']
    for cond in conditions:
        r = res[cond]
        stats = {}
        for k in METRICS:
            stats[k] = (macro(r['no_change'][k], names), macro(r['shared_text'][k], names),
                        bootstrap(r['shared_text'][k], r['no_change'][k], names))
        mc = [np.mean([d[side]['mean_cosine'] for d in analysis['diagnostics'][cond]]) for side in ('image', 'text')]
        f = lambda k: f'{stats[k][2][0]:+.4f} [{stats[k][2][1]:+.4f}, {stats[k][2][2]:+.4f}]'
        lines.append(f"| {cond} | {stats['ap'][0]:.4f} | {stats['ap'][1]:.4f} | {f('ap')} | {stats['noun'][0]:.3f} | "
                     f"{stats['noun'][1]:.3f} | {f('noun')} | {stats['joint'][0]:.3f} | {stats['joint'][1]:.3f} | {mc[0]:.3f} / {mc[1]:.3f} |")
        for method in METHODS:
            rows.append(dict(condition=cond, method=method, **{k: macro(r[method][k], names) for k in METRICS}))
    lines += ['', '## All methods under the reference condition', '',
              '| Method | AP | First result has both | State | Category |', '|---|---:|---:|---:|---:|']
    ref = res[conditions[0]]
    for method in METHODS:
        lines.append(f"| {method} | {macro(ref[method]['ap'], names):.4f} | {macro(ref[method]['joint'], names):.3f} | "
                     f"{macro(ref[method]['state'], names):.3f} | {macro(ref[method]['noun'], names):.3f} |")
    lines += ['', '## Limits', '',
        'Exploratory. Same data for all conditions; folds share categories and are not independent replications; the sample is a fresh draw, so numbers are not expected to match any earlier sample exactly, only the pattern (shared direction above no change, reversed below, category retention below no change). Layers were chosen in an earlier, unrelated study. A condition chosen after seeing these results would need a fresh evaluation. Diagnostics describe geometry and do not show cause.', '',
        'Protocol: STATE_CONTROLS_PROTOCOL.md. Per-category values and chosen step sizes: results.json.']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (out / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'method'] + METRICS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=Path('data/states'))
    parser.add_argument('--out', type=Path, default=Path('results/state_controls'))
    parser.add_argument('--random-controls', type=int, default=100)
    args = parser.parse_args(argv)
    data, manifest = load_run(args.run)
    names = [r['name'] for r in data['relations']]
    analysis = analyze(data, random_controls=args.random_controls)
    write_json(args.out / 'results.json', dict(manifest_hash=manifest['hash'], conditions=CONDITIONS, **analysis))
    report(analysis, names, args.out, CONDITIONS)
    print((args.out / 'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()