"""Rough sensitivity planning from four retired pilot rotations, not a power guarantee."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from experiment import write_json

run=Path('results/semantic_pilot')
paths=[run/'forward/results.json']+[run/f'rotation_{i}/results.json' for i in [1,2,3]]
rows=[json.loads(p.read_text()) for p in paths]
contrasts=['text_minus_no_shift','text_minus_random','text_minus_reverse']
result={'warning':'Four pilot categories with overlapping fits; PCA8 rather than main PCA32. Noncentral-normal approximation ignores fitted-map uncertainty. Not an adequate power guarantee.',
        'categories':[r['categories'][0] for r in rows], 'contrasts':{}}
lines=['# Retired pilot sensitivity planning','','Four rotating held-out pilot categories, with two map/direction categories and one validation category per rotation. The fitted maps overlap; PCA8 differs from main PCA32. These estimates are unstable and are not evidence that the main study is adequately powered.','','| Contrast | Pilot category effects | Category SD | Approximate MDE at 12 categories |','|---|---|---:|---:|']
for name in contrasts:
    values=np.array([r['inference'][name]['effect'] for r in rows]);sd=values.std(ddof=1)
    curve={str(n):float((norm.ppf(1-.05/6)+norm.ppf(.8))*sd/np.sqrt(n)) for n in [4,8,12,24,48]}
    result['contrasts'][name]={'pilot_effects':values.tolist(),'sd':float(sd),'normal_approx_80pct_power_mde':curve}
    lines.append(f'| {name} | {values.round(4).tolist()} | {sd:.4f} | {curve["12"]:.4f} |')
lines+=['','MDE assumes a normally approximated category mean, 80% power and three Bonferroni contrasts. The .02 meaningful-effect reference is a prespecified interpretation threshold, not a validated detectable effect. Report main interval widths and expand categories if precision is insufficient. Do not treat the four rotations as independent replications.']
write_json(run/'planning.json',result);(run/'PLANNING.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
