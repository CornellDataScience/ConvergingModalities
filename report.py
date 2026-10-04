"""Build a descriptive reduced-protocol report, never a paper-replication verdict."""
from experiment import load_manifest, digest
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/baseline'))
    args = parser.parse_args()
    manifest = load_manifest(args.run)
    alignment = json.loads((args.run / 'alignment.json').read_text())
    if alignment['manifest_hash'] != digest(manifest):
        raise ValueError('Alignment output belongs to a different manifest')
    language = json.loads((args.run / 'language_scores.json').read_text())
    performance = {r['identity']['model']['id']: r for r in language}
    lines = ['# Reduced Mac experiment', '',
             '**Status: reduced-protocol experiment; exact paper reproduction is not established.**', '',
             f"Sample: {manifest['samples']} original WIT image–caption pairs; k={manifest['config']['k']}.",
             'Language performance: 64 OpenWebText documents, capped at 1024 characters each; EOS conditioning and fixed windows. This differs from the paper’s 4M-token evaluation.', '',
             '| Language model | Vision model | Alignment | Selected layers (zero-based) | Shuffled max mean | Shuffled max 95th percentile | 1 − BPB |',
             '|---|---|---:|---|---:|---:|---:|']
    fig, ax = plt.subplots(figsize=(8, 5))
    summaries = []
    for vision in sorted({r['vision'] for r in alignment['results']}):
        rows = [r for r in alignment['results'] if r['vision'] == vision]
        rows.sort(key=lambda r: performance[r['language']]['one_minus_bpb'])
        xs = [performance[r['language']]['one_minus_bpb'] for r in rows]
        ys = [r['best_score'] for r in rows]
        ax.plot(xs, ys, 'o-', label=vision.split('/')[1].split('.')[0])
        for x, y, r in zip(xs, ys, rows):
            ax.annotate(r['language'].split('/')[1], (x, y), xytext=(3, 5), textcoords='offset points', fontsize=8)
            null = r['permuted_maxima']
            lines.append(f"| {r['language']} | {vision} | {y:.6f} | {r['best_layers']} | {np.mean(null):.6f} | {np.quantile(null,.95):.6f} | {x:.6f} |")
        if len(xs) > 1:
            correlation = np.corrcoef(xs, ys)[0, 1]
            summary = (f'{vision}: descriptive Pearson r={correlation:.4f} (n={len(xs)}); '
                       f'alignment strictly increases with measured language performance: {bool(np.all(np.diff(ys) > 0))}.')
            print(summary)
            summaries.append(summary)
    ax.set_xlabel('1 − bits per byte (reduced OpenWebText protocol)')
    ax.set_ylabel('Maximum mutual k-NN alignment')
    ax.set_title('Mac-scale experiment: original WIT sample, reduced model roster')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.run / 'trend.png', dpi=160)
    plt.close(fig)
    lines.extend(['', '## Observed trends', ''] + ['- ' + s for s in summaries])
    lines.extend(['', '## Interpretation limits', '',
                  '- Three related language checkpoints do not establish a cross-family or universal trend.',
                  '- Layer maxima and all layer-pair scores are preserved; permutation controls repeat the entire layer search.',
                  '- No matched numerical reference arrays have been obtained, so no numerical replication verdict is warranted.',
                  '- BLOOMZ versus BLOOM naming, concatenation, clipping, precision, and language evaluation remain documented differences from the paper.',
                  '- These models were evaluated, not retrained.', '', '## Provenance', '',
                  f"Upstream commit: `{manifest['upstream_commit']}`.",
                  f"Dataset revision: `{manifest['dataset_revision']}`.",
                  f"Sample hash: `{manifest['sample_hash']}`.",
                  'See manifest.json, feature metadata, language_corpus.json, and requirements.lock.txt for the complete configuration.', ''])
    (args.run / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
