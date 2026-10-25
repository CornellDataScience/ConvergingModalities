"""Between-condition comparison of state-transfer results (descriptive).

For each condition, compares (shared direction - no change) with the same quantity under the
reference condition, using the per-category values saved by state_transfer.py.

    python state_compare.py --results results/state_controls/results.json
"""
import argparse
import json
from pathlib import Path
from state_transfer import bootstrap, macro

METRIC_LABELS = {'ap': 'AP gain', 'noun': 'Category-retention change', 'joint': 'Joint first-result gain'}


def compare(results, conditions, names, reference='reference', draws=2000):
    rows = []
    for cond in conditions:
        if cond == reference:
            continue
        row = {'condition': cond}
        for metric in METRIC_LABELS:
            def gain(c):
                return {k: results[c]['shared_text'][metric][k] - results[c]['no_change'][metric][k]
                        for k in results[c]['shared_text'][metric]}
            row[metric] = bootstrap(gain(cond), gain(reference), names, draws=draws)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/state_controls/results.json'))
    args = parser.parse_args(argv)
    saved = json.loads(args.results.read_text(encoding='utf-8'))
    results, conditions = saved['results'], saved['conditions']
    names = sorted({k.split('|')[0] for k in results[conditions[0]]['no_change']['ap']})
    rows = compare(results, conditions, names)
    lines = ['# Between-condition differences (state transfer)', '',
             'Each cell is (effect under the condition) minus (effect under `reference`), with a category-label bootstrap 95% interval (2,000 draws). Descriptive and unadjusted for multiple comparisons; conditions share data, so differences are strongly dependent. An interval containing 0 means no detectable difference from the reference here.', '',
             '| Condition | ' + ' | '.join(METRIC_LABELS.values()) + ' |', '|---|---|---|---|']
    for row in rows:
        cells = [f'{row[m][0]:+.4f} [{row[m][1]:+.4f}, {row[m][2]:+.4f}]' for m in METRIC_LABELS]
        lines.append(f"| {row['condition']} | " + ' | '.join(cells) + ' |')
    out = args.results.with_name('REPORT_COMPARE.md')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
