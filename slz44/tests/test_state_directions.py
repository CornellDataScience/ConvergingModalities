import numpy as np
from state_directions import analyze, cosine_matrix, match_stats, summarize
from test_state_transfer import synthetic


def test_match_stats_identity_and_chance():
    stats = match_stats(np.eye(5))
    assert stats['margin'] == 1.0 and stats['top1'] == 1.0
    assert match_stats(np.ones((5, 5)))['margin'] == 0.0


def test_cosine_matrix_range():
    rng = np.random.default_rng(0)
    m = cosine_matrix(rng.standard_normal((4, 9)), rng.standard_normal((4, 9)))
    assert m.shape == (4, 4) and np.all(np.abs(m) <= 1 + 1e-9)


def test_planted_shift_is_recovered_and_beats_shuffled_bridge():
    folds = analyze(synthetic(), ['reference', 'center_only'], shuffles=20, splits=5)
    for cond, runs in folds.items():
        s = summarize(runs)
        assert s['text_top1'] > 0.8
        assert s['text_margin'] > s['null_margin_q95']
        assert s['image_margin'] > 0


def test_category_records_bootstrap_and_posthoc():
    from state_directions import bootstrap_categories
    data = synthetic()
    folds = analyze(data, ['reference'], shuffles=5, splits=3)['reference']
    records = [r for f in folds for r in f['records']]
    assert len(records) == sum(len(v) for v in data['nouns_for'].values())
    s = summarize(folds, draws=200)
    cat = s['category']
    assert cat['top1']['low'] <= cat['top1']['mean'] <= cat['top1']['high']
    assert cat['top1']['low'] > 0.2
    assert -2 <= s['posthoc_margin'] <= 2 and 0 <= s['posthoc_top1'] <= 1
    again = bootstrap_categories(records, 200)
    assert again == bootstrap_categories(records, 200)
