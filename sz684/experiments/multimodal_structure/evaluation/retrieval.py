"""Relation retrieval retains object/event identity as an explicit guardrail."""
import numpy as np
from .rsa import normalize_rows


def retrieve(features, records, direction, scale):
    """Shift source queries and rank all endpoints, excluding the same source.

    Records must describe one relation. Labels are 0=source, 1=target. A relevant
    result has both target state and the query's family. These are independent
    examples, not verified before/after instances.
    """
    bank = normalize_rows(features)
    values = []
    for i, row in enumerate(records):
        if str(row['label']) != '0':
            continue
        allowed = np.array([other['source_id'] != row['source_id'] for other in records])
        relevant = np.array([str(other['label']) == '1' and other['family_id'] == row['family_id']
                             for other in records]) & allowed
        if not relevant.any():
            continue
        query = normalize_rows((features[i] + scale*direction)[None, :])[0]
        order = np.argsort(-(bank @ query), kind='stable')
        order = order[allowed[order]]
        truth = relevant[order]
        precision = truth.cumsum()/np.arange(1, len(truth)+1)
        winner = records[order[0]]
        values.append({'family': row['family_id'], 'ap': float(np.sum(precision*truth)/truth.sum()),
                       'identity': float(winner['family_id'] == row['family_id']),
                       'joint': float(winner['family_id'] == row['family_id'] and str(winner['label']) == '1')})
    if not values:
        raise ValueError('No evaluable source queries with independent relevant targets')
    families = sorted({row['family'] for row in values})
    return {key: float(np.mean([np.mean([r[key] for r in values if r['family'] == family])
                               for family in families])) for key in ['ap','identity','joint']}


def select_scale(features, records, direction, scales):
    """Use validation AP only, choosing the first (smallest) scale on ties."""
    if list(scales) != sorted(scales) or min(scales) < 0:
        raise ValueError('Scales must be nonnegative and ascending')
    scores = [retrieve(features, records, direction, scale)['ap'] for scale in scales]
    return scales[int(np.argmax(scores))]
