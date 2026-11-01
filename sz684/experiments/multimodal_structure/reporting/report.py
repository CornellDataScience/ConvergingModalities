"""Readable report from saved scores; the report never reruns model fitting."""
import json
from pathlib import Path


def report(results_path, output):
    """Keep limitations next to scores so pilot findings cannot look confirmatory."""
    results = json.loads(Path(results_path).read_text())
    lines = ['# Cross-modal geometry pilot', '', results['status'], '',
             str(results.get('dataset_status', '')), '',
             f"Split: {results['split']}; common concepts: {len(results['concepts'])}; k: {results['k']}.", '',
             '| Model pair | RSA | Linear CKA | Jaccard | Neighbor overlap |',
             '| --- | ---: | ---: | ---: | ---: |']
    for pair in results['pairs']:
        values = pair['metrics']
        lines.append(f"| {pair['left']} / {pair['right']} | " +
                     ' | '.join(f"{values[key]['score']:.4f}" for key in ['rsa','cka','jaccard','overlap']) + ' |')
    lines += ['', '## Interpretation', '',
              'Scores describe concept geometry in frozen features, not semantic equivalence.',
              'Permutation p-values and full null draws are saved in the JSON. They are unadjusted and assume exchangeable concepts.',
              'Related concepts, weak labels, shared pretraining data, and source context may explain agreement.',
              'No encoder-training replication, independent label verification, or confirmatory significance is established by this pilot.',
              'RSA/CKA/neighbor metrics do not establish successful directional transfer.']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text('\n'.join(lines)+'\n')
