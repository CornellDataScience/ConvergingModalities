"""Convert existing slz44 outputs into the shared results format (schema 0.1).

    python tools/convert_slz44.py scale-grid results/scale_grid/results.json SCALE_GRID_PROTOCOL.md --out out.json
    python tools/convert_slz44.py categories results/state_directions/categories.csv STATE_DIRECTIONS_PROTOCOL.md --out out.json
"""
import argparse
import csv
import json
from pathlib import Path
import prh

SCALE_METRICS = {'knn_fixed': 'knn_excess_fixed_k', 'knn_scaled': 'knn_excess_scaled_k',
                 'cka': 'cka_excess', 'gw': 'gw_cost_reduction'}
CATEGORY_METRICS = {'cos_own': 'state_direction_cos_own', 'margin': 'state_direction_margin',
                    'top1': 'state_direction_top1'}


def scale_grid(results, protocol, run_id='scale-grid-1', commit=None):
    rows = results['rows']
    models = [{'id': m, 'modality': 'language'} for m in sorted({r['language'] for r in rows})]
    models += [{'id': m, 'modality': 'vision'} for m in sorted({r['vision'] for r in rows})]
    doc = prh.new_run(run_id, 'shared', protocol, dataset={'name': 'WIT', 'n_pool': 1024}, models=models,
                      commit=commit, notes='Converted from scale_grid results.json')
    for r in rows:
        for key, metric in SCALE_METRICS.items():
            extra = {}
            k = 10 if key == 'knn_fixed' else (r['k_scaled'] if key == 'knn_scaled' else None)
            fields = dict(n=r['n'], seed=r['seed'], split=f"seed-{r['seed']}",
                          language_model=r['language'], vision_model=r['vision'])
            if k is not None:
                fields['k'] = int(k)
            prh.add_record(doc, metric, r['mode'], float(r[key]), **fields, **extra)
    return doc


def categories(rows, protocol, run_id='state-directions-1', commit=None,
               models=(('BLOOMZ-1.7B', 'language'), ('DINOv2-base', 'vision'))):
    doc = prh.new_run(run_id, '1', protocol, dataset={'name': 'MIT-States'},
                      models=[{'id': i, 'modality': m} for i, m in models], commit=commit,
                      notes='Converted from categories.csv')
    for r in rows:
        for key, metric in CATEGORY_METRICS.items():
            prh.add_record(doc, metric, r['condition'], float(r[key]), fold=int(r['fold']),
                           relation=r['relation'], noun=r['noun'])
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['scale-grid', 'categories'])
    parser.add_argument('source', type=Path)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.kind == 'scale-grid':
        doc = scale_grid(json.loads(args.source.read_text(encoding='utf-8')), args.protocol)
    else:
        with args.source.open(newline='', encoding='utf-8') as f:
            doc = categories(list(csv.DictReader(f)), args.protocol)
    prh.write(doc, args.out)
    print(f'Wrote {len(doc["records"])} records to {args.out}')


if __name__ == '__main__':
    main()
