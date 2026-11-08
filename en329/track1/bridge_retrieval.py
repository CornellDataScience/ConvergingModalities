"""
Track 1 / M2: bridge controls, held-out-family retrieval, alignment vs retrieval gain.
See PROTOCOL.md for the frozen design. Numpy/scipy only.

Inputs (see PROTOCOL.md for the full contract):
    bridge.npz   text (n, d_t), image (n, d_i)   paired rows that fit the bridge
    states.npz   image_X, image_object, image_state, text_X, text_object, text_state
    split.json   {"train": [...], "val": [...], "test": [...],
                  "transitions": [{"name": "cooking", "source": "raw", "target": "cooked"}, ...]}

Usage:
    python bridge_retrieval.py --bridge bridge.npz --states states.npz --split split.json --out results/run
    python bridge_retrieval.py --synthetic --out results/synthetic            # controls should pass
    python bridge_retrieval.py --synthetic --null-text --out results/null     # bridge should NOT beat shuffled
"""
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
METRICS = ('joint_rr', 'joint_at1', 'joint_at5', 'object_at1', 'state_at1')
PRIMARY = 'joint_rr'
CONFIRMATORY = [('bridge', 'no_change'), ('bridge', 'shuffled_bridge'),
                ('bridge', 'reversed_bridge'), ('bridge', 'other_transition')]
DESCRIPTIVE = [('image_direction', 'bridge')]
SAME_STEP = ('reversed_bridge', 'other_transition')  # wrong-direction controls reuse the bridge's validated step
CALIBRATION_COS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0)
MIN_TEST_QUERIES = 10


# ---------- preprocessing and the bridge (same choices as sz684/difference_transfer.py) ----------
def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fit_pca(x, dim=128):
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(dim, x.shape[1], x.shape[0] - 1)
    return mean, vt[:k].T


def apply_pca(x, fitted):
    mean, basis = fitted
    return normalize((x - mean) @ basis)


def ridge_solver(x, alpha=1.0):
    """Returns S such that the ridge map for targets y is S @ (y - y.mean(0)); lets shuffles skip refitting."""
    xc = x - x.mean(0)
    return np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T)


def fit_ridge(x, y, alpha=1.0):
    return ridge_solver(x, alpha) @ (y - y.mean(0))


# ---------- directions ----------
def means_by(X, objects, states, keep=None):
    groups = {}
    for i, key in enumerate(zip(objects.tolist(), states.tolist())):
        if keep is None or key[0] in keep:
            groups.setdefault(key, []).append(i)
    return {k: X[v].mean(0) for k, v in groups.items()}


def eligible(objects, s, t, *tables):
    return [o for o in sorted(objects) if all((o, s) in m and (o, t) in m for m in tables)]


def direction(means, objects, s, t):
    return np.mean([means[o, t] - means[o, s] for o in objects], axis=0)


# ---------- retrieval ----------
def score(queries, gallery, g_obj, g_state, q_obj, q_target, q_self):
    """Rank the gallery for each query (excluding the query's own image); return per-query metrics."""
    sims = queries @ gallery.T
    sims[np.arange(len(q_self)), q_self] = -np.inf
    correct = (g_obj[None] == q_obj[:, None]) & (g_state[None] == q_target[:, None])
    correct[np.arange(len(q_self)), q_self] = False
    best = np.where(correct, sims, -np.inf).max(1)
    has = np.isfinite(best)
    rank = 1 + (sims > best[:, None]).sum(1)
    top = sims.argmax(1)
    return dict(joint_rr=np.where(has, 1.0 / rank, 0.0),
                joint_at1=(has & (rank == 1)).astype(float),
                joint_at5=(has & (rank <= 5)).astype(float),
                object_at1=(g_obj[top] == q_obj).astype(float),
                state_at1=(g_state[top] == q_target).astype(float),
                rank=np.where(has, rank, -1), top1=top)


def make_partition(image, obj, state, objects, transitions):
    idx = np.flatnonzero(np.isin(obj, list(objects)))
    g_obj, g_state = obj[idx], state[idx]
    have = set(zip(g_obj.tolist(), g_state.tolist()))
    queries = {}
    for tr in transitions:
        s, t = tr['source'], tr['target']
        q = [i for i in range(len(idx)) if g_state[i] == s and (g_obj[i], t) in have]
        queries[tr['name']] = np.array(q, dtype=int)
    return dict(index=idx, X=image[idx], obj=g_obj, state=g_state, queries=queries)


def evaluate(part, transitions, dirs, alpha):
    """dirs: {transition name: unit vector} or None for no change. Returns concatenated per-query arrays."""
    out = {m: [] for m in METRICS + ('rank', 'top1')}
    out.update(transition=[], object=[], query=[])
    for tr in transitions:
        q = part['queries'][tr['name']]
        if len(q) == 0:
            continue
        x = part['X'][q]
        if dirs is not None:
            x = normalize(x + alpha * dirs[tr['name']])
        r = score(x, part['X'], part['obj'], part['state'], part['obj'][q],
                  np.full(len(q), tr['target']), q)
        for m in METRICS + ('rank',):
            out[m].append(r[m])
        out['top1'].append(part['index'][r['top1']])
        out['query'].append(part['index'][q])
        out['transition'].append(np.full(len(q), tr['name']))
        out['object'].append(part['obj'][q])
    return {k: np.concatenate(v) for k, v in out.items()}


def choose_alpha(val, transitions, dirs):
    scores = [evaluate(val, transitions, dirs, a)[PRIMARY].mean() for a in ALPHAS]
    return ALPHAS[int(np.argmax(scores))], dict(zip(map(str, ALPHAS), map(float, scores)))


# ---------- inference ----------
def cluster_bootstrap(diff, clusters, draws=10000, seed=0, family=1):
    ids, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(draws, len(ids)))
    est = sums[pick].sum(1) / counts[pick].sum(1)
    adj = 0.05 / family
    return dict(effect=float(diff.mean()), clusters=int(len(ids)), queries=int(len(diff)),
                ci95=[float(np.quantile(est, .025)), float(np.quantile(est, .975))],
                ci_adjusted=[float(np.quantile(est, adj / 2)), float(np.quantile(est, 1 - adj / 2))],
                family_size=family)


def exact_spearman(x, y):
    rho = spearmanr(x, y).statistic
    if len(x) > 8:
        raise ValueError('exact permutation only for small n')
    perms = [spearmanr(x, np.array(p)).statistic for p in itertools.permutations(y)]
    p = float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))
    return float(rho), p


# ---------- experiment ----------
def run(data, split, shuffles=200, calibration_draws=20, seed=0, pca_dim=128, bootstrap_draws=10000):
    train, val, test = (set(split[k]) for k in ('train', 'val', 'test'))
    assert not (train & val or train & test or val & test), 'object splits must be disjoint'
    transitions = split['transitions']
    names = [tr['name'] for tr in transitions]
    rng = np.random.default_rng(seed)

    tp, ip = fit_pca(normalize(data['bridge_text']), pca_dim), fit_pca(normalize(data['bridge_image']), pca_dim)
    bt, bi = apply_pca(normalize(data['bridge_text']), tp), apply_pca(normalize(data['bridge_image']), ip)
    text = apply_pca(normalize(data['text_X']), tp)
    image = apply_pca(normalize(data['image_X']), ip)
    obj, state = data['image_object'], data['image_state']

    solver = ridge_solver(bt)
    bic = bi - bi.mean(0)
    W = solver @ bic
    tmeans = means_by(text, data['text_object'], data['text_state'])
    imeans = means_by(image, obj, state, keep=train)

    text_dir, vis_dir, halves, train_objects = {}, {}, {}, {}
    for tr in transitions:
        s, t, n = tr['source'], tr['target'], tr['name']
        objs = eligible(train, s, t, tmeans, imeans)
        if len(objs) < 2:
            raise ValueError(f'{n}: only {len(objs)} eligible training objects')
        train_objects[n] = objs
        text_dir[n] = direction(tmeans, objs, s, t)
        vis_dir[n] = normalize(direction(imeans, objs, s, t))
        perm = rng.permutation(objs).tolist()
        halves[n] = (perm[:len(objs) // 2], perm[len(objs) // 2:])

    def mapped(td, Wm):
        return {n: normalize(td[n] @ Wm) for n in td}

    bridge = mapped(text_dir, W)
    conditions = {
        'no_change': None,
        'bridge': bridge,
        'reversed_bridge': {n: -v for n, v in bridge.items()},
        'other_transition': {n: normalize(np.mean([bridge[m] for m in names if m != n], 0)) for n in names},
        'subset_a': mapped({tr['name']: direction(tmeans, halves[tr['name']][0], tr['source'], tr['target'])
                            for tr in transitions}, W),
        'subset_b': mapped({tr['name']: direction(tmeans, halves[tr['name']][1], tr['source'], tr['target'])
                            for tr in transitions}, W),
        'image_direction': vis_dir,
    }

    valp = make_partition(image, obj, state, val, transitions)
    testp = make_partition(image, obj, state, test, transitions)
    if sum(len(q) for q in valp['queries'].values()) == 0:
        raise ValueError('no validation queries; alpha cannot be chosen')

    alphas, val_curves, test_rows = {}, {}, {}
    for name, dirs in conditions.items():
        if dirs is None:
            alphas[name] = 0.0
        elif name in SAME_STEP:
            alphas[name] = alphas['bridge']
        else:
            alphas[name], val_curves[name] = choose_alpha(valp, transitions, dirs)
        test_rows[name] = evaluate(testp, transitions, dirs, alphas[name])

    shuffle_alphas, shuffle_means, shuffle_margins = [], [], []
    shuffle_sum = {m: 0.0 for m in METRICS}
    for _ in range(shuffles):
        Ws = solver @ bic[rng.permutation(len(bic))]
        dirs = mapped(text_dir, Ws)
        a, _ = choose_alpha(valp, transitions, dirs)
        r = evaluate(testp, transitions, dirs, a)
        shuffle_alphas.append(a)
        shuffle_means.append(float(r[PRIMARY].mean()))
        for m in METRICS:
            shuffle_sum[m] = shuffle_sum[m] + r[m]
        shuffle_margins.append(alignment(dirs, vis_dir, names)['margin_mean'])
    shuffled = dict(test_rows['bridge'])
    shuffled.update({m: shuffle_sum[m] / shuffles for m in METRICS})
    test_rows['shuffled_bridge'] = shuffled

    groups = test_rows['bridge']['transition']
    summary = {}
    for name, rows in test_rows.items():
        summary[name] = dict(alpha=alphas.get(name), overall={m: float(rows[m].mean()) for m in METRICS},
                             per_transition={n: {m: float(rows[m][groups == n].mean()) if (groups == n).any() else None
                                                 for m in METRICS} for n in names})
    summary['shuffled_bridge']['alpha'] = dict(values=list(map(float, shuffle_alphas)))
    observed = summary['bridge']['overall'][PRIMARY]
    shuffle_tail = (1 + sum(v >= observed for v in shuffle_means)) / (1 + shuffles)

    clusters = test_rows['bridge']['object']
    contrasts = []
    for i, (a, b) in enumerate(CONFIRMATORY + DESCRIPTIVE):
        family = len(CONFIRMATORY) if (a, b) in CONFIRMATORY else 1
        d = test_rows[a][PRIMARY] - test_rows[b][PRIMARY]
        r = cluster_bootstrap(d, clusters, bootstrap_draws, seed + 100 + i, family)
        r.update(contrast=f'{a} - {b}', confirmatory=family > 1,
                 object_retention_difference=float((test_rows[a]['object_at1'] - test_rows[b]['object_at1']).mean()))
        contrasts.append(r)

    # Query bootstraps hold the fitted bridge fixed, so they miss the bridge's own randomness; the draw-level
    # shuffle tail covers that. A positive verdict needs both.
    passes = all(c['ci_adjusted'][0] > 0 for c in contrasts if c['confirmatory'])
    verdict = dict(confirmatory_cis_exclude_zero=passes, shuffle_tail_at_most_0_05=bool(shuffle_tail <= 0.05),
                   transfer_supported=bool(passes and shuffle_tail <= 0.05))

    align = alignment(bridge, vis_dir, names)
    pt = lambda c, n: summary[c]['per_transition'][n][PRIMARY]
    gains_by = {n: pt('bridge', n) - pt('no_change', n) for n in names if pt('bridge', n) is not None}
    usable = [i for i, n in enumerate(names) if n in gains_by]
    across = dict(transitions=[names[i] for i in usable], margin=[align['margin'][i] for i in usable],
                  gain=[gains_by[names[i]] for i in usable])
    if len(usable) >= 3:
        across['spearman_rho'], across['exact_permutation_p'] = exact_spearman(
            np.array(across['margin']), np.array(across['gain']))

    # Calibration: random directions at a fixed cosine to the training visual direction, at the bridge's step.
    # Any direction closer to the visual one retrieves better, so alignment -> gain holds even when text is
    # uninformative. The question is whether the bridge gains more than its alignment alone predicts.
    base = {n: test_rows['no_change'][PRIMARY][groups == n].mean() for n in names if (groups == n).any()}
    calibration = {}
    for i, tr in enumerate(transitions):
        n, v = tr['name'], vis_dir[tr['name']]
        if n not in base:
            continue
        curve = []
        for c in CALIBRATION_COS:
            gains = []
            for _ in range(calibration_draws):
                r = rng.normal(size=v.shape)
                d = c * v + np.sqrt(1 - c * c) * normalize(r - (r @ v) * v)
                gains.append(evaluate(testp, [tr], {n: d}, alphas['bridge'])[PRIMARY].mean() - base[n])
            curve.append(dict(cos=c, mean=float(np.mean(gains)), q05=float(np.quantile(gains, .05)),
                              q95=float(np.quantile(gains, .95))))
        a = align['matrix'][i][i]
        at = lambda key: float(np.interp(a, CALIBRATION_COS, [p[key] for p in curve]))
        calibration[n] = dict(curve=curve, bridge_alignment=a, bridge_gain=float(gains_by[n]),
                              expected_gain=at('mean'), band90=[at('q05'), at('q95')],
                              residual=float(gains_by[n]) - at('mean'))

    stability = {n: dict(text_halves_cos=float(conditions['subset_a'][n] @ conditions['subset_b'][n]),
                         train_objects=len(train_objects[n])) for n in names}
    query_counts = {n: dict(val=int(len(valp['queries'][n])), test=int(len(testp['queries'][n])),
                            low=bool(len(testp['queries'][n]) < MIN_TEST_QUERIES)) for n in names}

    keep = ['no_change', 'bridge', 'reversed_bridge', 'other_transition', 'subset_a', 'subset_b', 'image_direction']
    per_query = dict(query=test_rows['bridge']['query'].tolist(), object=clusters.tolist(),
                     transition=groups.tolist(),
                     conditions={c: {k: test_rows[c][k].tolist() for k in METRICS + ('rank', 'top1')} for c in keep})
    return dict(schema='convmod.track1.m2/draft-1', primary_metric=PRIMARY, transitions=transitions,
                config=dict(shuffles=shuffles, calibration_draws=calibration_draws, seed=seed, pca_dim=pca_dim, ridge_alpha=1.0,
                            alpha_grid=list(ALPHAS), bootstrap_draws=bootstrap_draws,
                            bridge_rows=int(len(bt)), pca_dims=[int(tp[1].shape[1]), int(ip[1].shape[1])]),
                query_counts=query_counts, validation_curves=val_curves, summary=summary,
                shuffled_bridge_null=dict(test_means=shuffle_means, tail_fraction_diagnostic=float(shuffle_tail)),
                contrasts=contrasts, verdict=verdict,
                alignment=dict(matrix=align['matrix'], rows='mapped text direction', cols='visual direction',
                               margin=dict(zip(names, align['margin'])), margin_mean=align['margin_mean'],
                               shuffled_margin_mean=float(np.mean(shuffle_margins)),
                               shuffled_margin_q95=float(np.quantile(shuffle_margins, .95))),
                direction_stability=stability,
                alignment_vs_gain=dict(across_transitions=across, calibration=calibration),
                per_query=per_query)


def alignment(mapped_dirs, vis_dir, names):
    A = np.array([[float(mapped_dirs[i] @ vis_dir[j]) for j in names] for i in names])
    off = (A.sum(1) - np.diag(A)) / max(len(names) - 1, 1)
    margin = (np.diag(A) - off).tolist()
    return dict(matrix=A.tolist(), margin=margin, margin_mean=float(np.mean(margin)))


# ---------- report ----------
def report(res):
    f = lambda v: 'n/a' if v is None else f'{v:.4f}'
    names = [t['name'] for t in res['transitions']]
    L = ['# Track 1 / M2: controlled retrieval comparison', '',
         f"Primary metric: `{res['primary_metric']}` (reciprocal rank of the first same-object, target-state image). "
         f"Bridge fitted on {res['config']['bridge_rows']} pairs, PCA dims (text, image) = {res['config']['pca_dims']}. "
         f"{res['config']['shuffles']} shuffled-bridge draws. λ chosen per condition on validation objects.", '',
         '## Test results by condition', '',
         '| Condition | λ | joint RR | joint@1 | joint@5 | object@1 (retention) | state@1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for c, s in res['summary'].items():
        a = s['alpha']
        a = 'per draw' if isinstance(a, dict) else f'{a:g}'
        o = s['overall']
        L.append(f"| {c} | {a} | {f(o['joint_rr'])} | {f(o['joint_at1'])} | {f(o['joint_at5'])} | {f(o['object_at1'])} | {f(o['state_at1'])} |")
    L += ['', '## Paired contrasts (cluster bootstrap over source objects)', '',
          '| Contrast | Δ joint RR | 95% CI | Adjusted CI | Excludes 0 (adjusted) | Δ object@1 | Objects / queries |',
          '|---|---:|---|---|---|---:|---|']
    for c in res['contrasts']:
        lo, hi = c['ci_adjusted']
        tag = ('yes' if lo > 0 or hi < 0 else 'no') if c['confirmatory'] else 'descriptive'
        L.append(f"| {c['contrast']} | {c['effect']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                 f"[{lo:+.4f}, {hi:+.4f}] | {tag} | {c['object_retention_difference']:+.4f} | {c['clusters']} / {c['queries']} |")
    v = res['verdict']
    L += ['', f"Shuffled-bridge tail fraction (diagnostic only): {res['shuffled_bridge_null']['tail_fraction_diagnostic']:.4f}", '',
          f"**Verdict: transfer {'supported' if v['transfer_supported'] else 'NOT supported'}** "
          f"(all confirmatory adjusted CIs above 0: {v['confirmatory_cis_exclude_zero']}; shuffle tail ≤ 0.05: {v['shuffle_tail_at_most_0_05']}).", '',
          '## Per-transition joint RR', '', '| Transition | test queries | no_change | bridge | shuffled | other_transition | image_direction |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        q = res['query_counts'][n]
        g = lambda c: f(res['summary'][c]['per_transition'][n]['joint_rr'])
        L.append(f"| {n}{' (low n)' if q['low'] else ''} | {q['test']} | {g('no_change')} | {g('bridge')} | {g('shuffled_bridge')} | {g('other_transition')} | {g('image_direction')} |")
    al = res['alignment']
    L += ['', '## Alignment matrix (rows: mapped text direction, cols: visual direction; train objects)', '',
          '| | ' + ' | '.join(names) + ' | margin |', '|---|' + '---:|' * (len(names) + 1)]
    for n, row in zip(names, al['matrix']):
        L.append(f'| {n} | ' + ' | '.join(f'{v:+.3f}' for v in row) + f" | {al['margin'][n]:+.3f} |")
    L += ['', f"Mean margin {al['margin_mean']:+.4f}; shuffled-bridge mean margin {al['shuffled_margin_mean']:+.4f} (95th pct {al['shuffled_margin_q95']:+.4f}).", '',
          '## Does alignment predict retrieval gain?', '']
    ac = res['alignment_vs_gain']['across_transitions']
    if 'spearman_rho' in ac:
        L.append(f"Across transitions (n={len(ac['gain'])}, descriptive): Spearman ρ = {ac['spearman_rho']:+.3f}, exact permutation p = {ac['exact_permutation_p']:.3f}.")
    cal = res['alignment_vs_gain']['calibration']
    if cal:
        L += ['', 'Calibration against random directions with the same cosine to the visual direction, at the bridge step. '
              'A residual inside the band means alignment alone explains the bridge\'s gain.', '',
              '| Transition | bridge cos to visual | bridge gain | expected from alignment | 90% band | residual |',
              '|---|---:|---:|---:|---|---:|']
        for n, c in cal.items():
            L.append(f"| {n} | {c['bridge_alignment']:+.3f} | {c['bridge_gain']:+.4f} | {c['expected_gain']:+.4f} | "
                     f"[{c['band90'][0]:+.4f}, {c['band90'][1]:+.4f}] | {c['residual']:+.4f} |")
    L += ['', '## Direction stability (cos between text directions from disjoint halves of train objects)', '',
          ' '.join(f"{n}: {s['text_halves_cos']:+.3f} ({s['train_objects']} objs);" for n, s in res['direction_stability'].items()),
          '', 'Caveats: exploratory; shuffled-bridge tail is diagnostic (layers chosen with true WIT pairs); '
          'transitions with fewer than 10 test queries are flagged; see PROTOCOL.md for claim boundaries.']
    return '\n'.join(L) + '\n'


# ---------- synthetic world ----------
SYN_TRANSITIONS = [dict(name='cleaning', source='dirty', target='clean'), dict(name='cooking', source='raw', target='cooked'),
                   dict(name='ripening', source='unripe', target='ripe'), dict(name='thawing', source='frozen', target='thawed'),
                   dict(name='drying', source='wet', target='dry')]


def synthetic(seed=0, null_text=False, families=9, per_family=4, images=6, latent=24):
    """Shared latent world seen through a nonlinear 'vision model' and a linear 'language model'.
    null_text=True makes text independent of the world, so the bridge carries nothing."""
    rng = np.random.default_rng(seed)
    states = sorted({tr[k] for tr in SYN_TRANSITIONS for k in ('source', 'target')})
    state_vec = {s: rng.normal(size=latent) * 0.8 for s in states}
    img_map = rng.normal(size=(latent, 96)) / np.sqrt(latent)
    txt_map = rng.normal(size=(latent, 80)) / np.sqrt(latent)
    see = lambda z: np.tanh(z @ img_map) + rng.normal(size=(len(z), 96)) * 0.15
    say = (lambda z: rng.normal(size=(len(z), 80))) if null_text else (lambda z: z @ txt_map + rng.normal(size=(len(z), 80)) * 0.6)
    rows = dict(image_X=[], image_object=[], image_state=[], text_X=[], text_object=[], text_state=[])
    split = dict(train=[], val=[], test=[], transitions=SYN_TRANSITIONS)
    for f in range(families):
        fam = rng.normal(size=latent)
        part = 'train' if f < families - 4 else 'val' if f < families - 2 else 'test'
        for i in range(per_family):
            o = f'f{f}_obj{i}'
            split[part].append(o)
            ov = fam + rng.normal(size=latent) * 0.7
            for tr in rng.choice(SYN_TRANSITIONS, size=3, replace=False):
                for s in (tr['source'], tr['target']):
                    z = ov + state_vec[s] + rng.normal(size=latent) * 0.9
                    rows['image_X'].append(see(z + rng.normal(size=(images, latent)) * 0.5))
                    rows['image_object'] += [o] * images
                    rows['image_state'] += [s] * images
                    rows['text_X'].append(say(z[None]))
                    rows['text_object'].append(o)
                    rows['text_state'].append(s)
    data = {k: (np.concatenate(v) if k.endswith('_X') else np.array(v)) for k, v in rows.items()}
    zb = rng.normal(size=(300, latent)) * 1.2
    data['bridge_text'], data['bridge_image'] = say(zb), see(zb)
    return data, split


def load(bridge_path, states_path, split_path):
    b, s = np.load(bridge_path), np.load(states_path)
    data = {k: s[k] for k in ('image_X', 'image_object', 'image_state', 'text_X', 'text_object', 'text_state')}
    data['bridge_text'], data['bridge_image'] = b['text'], b['image']
    hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (bridge_path, states_path, split_path)}
    return data, json.loads(Path(split_path).read_text()), hashes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bridge'); p.add_argument('--states'); p.add_argument('--split')
    p.add_argument('--synthetic', action='store_true'); p.add_argument('--null-text', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--shuffles', type=int, default=200); p.add_argument('--calibration-draws', type=int, default=20)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--pca-dim', type=int, default=128)
    a = p.parse_args()
    if a.synthetic:
        data, split = synthetic(a.seed, a.null_text)
        hashes = dict(synthetic=dict(seed=a.seed, null_text=a.null_text))
    else:
        if not (a.bridge and a.states and a.split):
            p.error('--bridge, --states and --split are required unless --synthetic')
        data, split, hashes = load(a.bridge, a.states, a.split)
    res = run(data, split, a.shuffles, a.calibration_draws, a.seed, a.pca_dim)
    here = Path(__file__).resolve().parent
    hashes.update({f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in ('bridge_retrieval.py', 'PROTOCOL.md')})
    res['input_hashes'] = hashes
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    (out / 'REPORT.md').write_text(report(res))
    print(report(res))


if __name__ == '__main__':
    main()
