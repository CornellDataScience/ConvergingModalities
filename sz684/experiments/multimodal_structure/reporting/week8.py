"""Export the locked week-8 results, including compact tables and static plots."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def comparison(left,right,models):
    """Name cross-modal and within-modality comparisons consistently."""
    a,b=models[left]['model']['modality'],models[right]['model']['modality']
    if a==b: return 'within-'+a
    return '-'.join(sorted([a,b]))


def write_csv(path,rows):
    """Write small plain tables that can be inspected without Python."""
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def heatmaps(geometry,category_rows,output):
    """Plot effect sizes only; colors do not represent statistical significance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(geometry['models'])
    index={name:i for i,name in enumerate(names)}
    figure,axes=plt.subplots(2,2,figsize=(13,11))
    for axis,key in zip(axes.flat,['rsa','cka','jaccard','overlap']):
        matrix=np.eye(len(names))
        for pair in geometry['pairs']:
            i,j=index[pair['left']],index[pair['right']]
            matrix[i,j]=matrix[j,i]=pair['metrics'][key]['score']
        image=axis.imshow(matrix,vmin=-1 if key=='rsa' else 0,vmax=1,cmap='viridis')
        axis.set_xticks(range(len(names)),names,rotation=45,ha='right')
        axis.set_yticks(range(len(names)),names);axis.set_title(key)
        for i in range(len(names)):
            for j in range(len(names)): axis.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
        figure.colorbar(image,ax=axis,fraction=.04)
    figure.suptitle('Locked held-out concept geometry (raw scores, no significance claim)')
    figure.tight_layout();figure.savefig(output/'geometry_heatmaps.png',dpi=150);plt.close(figure)
    axis_names=sorted({row['axis'] for row in category_rows})
    figure,axes=plt.subplots(1,len(axis_names),figsize=(14,6),squeeze=False)
    for axis,name in zip(axes.flat,axis_names):
        matrix=np.full((len(names),len(names)),np.nan)
        for row in category_rows:
            if row['axis']==name:
                matrix[index[row['source']],index[row['target']]]=row['transfer_balanced_accuracy']
        image=axis.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
        axis.set_xticks(range(len(names)),names,rotation=45,ha='right')
        axis.set_yticks(range(len(names)),names);axis.set_title(name)
        axis.set_xlabel('Target model');axis.set_ylabel('Source model')
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(matrix[i,j]): axis.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
        figure.colorbar(image,ax=axis,fraction=.04)
    figure.suptitle('Transferred category-probe balanced accuracy (chance = 0.5)')
    figure.tight_layout();figure.savefig(output/'category_transfer.png',dpi=150);plt.close(figure)


def report_study(study_folder,output):
    """Read saved artifacts without fitting or rerunning a test evaluation."""
    study,output=Path(study_folder),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result=json.loads((study/'week8_test.json').read_text())
    geometry=json.loads((study/'week8_test_geometry.json').read_text())
    fits=json.loads((study/'week7_fits.json').read_text())
    audit=json.loads((study/'media_audit_final.json').read_text())
    plan=json.loads((study/'sampling_plan.json').read_text())
    rows=[]
    for pair in geometry['pairs']:
        row={'comparison':comparison(pair['left'],pair['right'],geometry['models']),
             'left':pair['left'],'right':pair['right']}
        row.update({key:score['score'] for key,score in pair['metrics'].items()})
        rows.append(row)
    category=[dict(comparison=comparison(row['source'],row['target'],geometry['models']),**row)
              for row in result['category_axis_results']]
    write_csv(output/'geometry_scores.csv',rows)
    write_csv(output/'category_scores.csv',category)
    labels=[]
    from ..data.axes import AXES,axis_label
    for label in plan['labels']:
        family=plan['family_by_label'][label]
        labels.append(dict(concept=label,family=family,split=plan['split_by_label'][label],
                           **{axis:axis_label(axis,family) for axis in AXES}))
    write_csv(output/'concepts_and_axes.csv',labels)
    heatmaps(geometry,category,output)
    lines=['# Week 8: locked held-out released-label results','',
           'Completed computational evaluation under the approved released-label scope. No encoder was fine-tuned.', '',
           f"Data: {len(plan['labels'])} concepts, {audit['by_modality']['audio']} paired recordings, "
           f"{audit['by_modality']['text']} text descriptions. Concept split: {audit['concepts_by_split']}.",
           f"Models: {len(geometry['models'])}; geometry pairs: {len(rows)}; ordered category-axis tests: {len(category)}.",
           '', '## Geometry on held-out concepts','',
           '| Pair | RSA | CKA | Jaccard | Top-3 overlap |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['left']} / {row['right']} | "+' | '.join(f"{row[key]:.4f}" for key in ['rsa','cka','jaccard','overlap'])+' |')
    lines+=['', '![Geometry scores](geometry_heatmaps.png)', '',
            f"Test concepts: {len(geometry['concepts'])}; random top-3 overlap expectation: {3/(len(geometry['concepts'])-1):.4f}.",
            'CKA and RSA have different scales and finite-sample behavior. Full correspondence nulls are saved; unadjusted p-values do not establish significance.',
            '', '## Category CAV transfer','',
            'The table averages fixed-fit scores across model directions in each comparison. It does not treat those directions as independent training replications.', '',
            '| Comparison | Axis | Maps | Transferred BA | Target probe BA | Shuffled anchors BA | Signed CAV cosine |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    grouped=defaultdict(list)
    for row in category: grouped[(row['comparison'],row['axis'])].append(row)
    for (kind,axis),values in sorted(grouped.items()):
        values_mean=[np.mean([row[key] for row in values]) for key in ['transfer_balanced_accuracy','target_balanced_accuracy','shuffled_anchor_accuracy','cav_cosine']]
        lines.append(f'| {kind} | {axis} | {len(values)} | '+' | '.join(f'{value:.4f}' for value in values_mean)+' |')
    lines+=['', '![Category transfer](category_transfer.png)', '',
            'BA means balanced accuracy; binary chance is 0.5. All individual scores, sample/class counts and selected dimensions are exported to category_scores.csv.',
            '', '## What was locked','',
            '- Sampling and backup order were saved before new feature extraction; first-pilot concepts and sources were excluded.',
            '- The initial provisional family plan and pre-extraction corrections remain saved. Three silent files were replaced without consulting model scores.',
            '- Training map-anchor sources and probe sources are disjoint. Test families do not fit transforms, maps, probes or hyperparameters.',
            '- Width and probe regularization were selected on validation only. Every fitted array and selection curve is saved.',
            '- Fit, data, cache, protocol and software hashes were locked before test scoring. Per-example predictions and per-concept neighborhoods are preserved.',
            '', '## Interpretation and remaining work','',
            '- Labels are released video-level AV annotations. The selected stills were decoded but not independently relabeled.',
            '- Category axes are explicit researcher groupings of released event names. They are not physical state changes; noun preservation is not a meaningful metric for these category classifiers.',
            '- Text inputs include event names and therefore provide explicit category cues. Cross-modal differences can reflect this information asymmetry.',
            '- Family grouping is provisional. Unique source IDs and exact-hash checks do not prove uploader independence or pretraining exclusion.',
            '- These are fixed-fit held-out scores. Refit confidence intervals, multiplicity-adjusted decisions and fresh-source replication remain for weeks 9–10.',
            '- Historical image–text object-state retrieval, including identity-retention results, remains a separate exploratory arm; no audiovisual state-change success is claimed.',
            '', '## Files','',
            '- `geometry_scores.csv`: all 15 unordered model pairs.',
            '- `category_scores.csv`: all 60 ordered category-axis comparisons, including controls and sample counts.',
            '- `concepts_and_axes.csv`: every concept, family, split and category-axis label.',
            '- Local run folder: sampling/QA logs, features, `week7_fits.json`, `week7_reliability.json`, `evaluation_lock.json`, `week8_test.json`, `week8_test_predictions.csv`, and geometry detail/null files.',
            '', f"Dataset identity: `{result['dataset_hash']}`.",f"Evaluation lock SHA256: `{result['lock_sha256']}`."]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return {'geometry_pairs':len(rows),'category_tests':len(category),'test_concepts':len(geometry['concepts'])}
