"""Explicit category axes derived from released event labels, not physical states.

These test concept classification and CAV transport. They do not define an
identity-preserving change to the same object and must not be scored as one.
"""
BIOLOGICAL_FAMILIES = {'birds','small_animals','felines','canines','human_vocal_and_body',
                       'other_cattle','other_donkey','other_elk','other_ferret','other_horse',
                       'other_sheep','other_chipmunk','other_gibbon','other_mouse','other_lip'}
ARTIFICIAL_FAMILIES = {'road_vehicles','rail','aircraft','watercraft','alarms','appliances','tools',
                       'other_blowtorch'}


def axis_label(axis, family):
    """Return 0/1, or None where the concept does not support this contrast."""
    if axis == 'instrument_performance':
        return int(family.startswith('instrument_'))
    if axis == 'biological_vs_manufactured':
        if family in BIOLOGICAL_FAMILIES: return 1
        if family in ARTIFICIAL_FAMILIES or family.startswith('instrument_'): return 0
        return None
    raise ValueError('Unknown category axis: '+axis)


AXES = ['instrument_performance','biological_vs_manufactured']


def covered_splits(rows):
    """Require both endpoint classes with several concepts in every split."""
    for split in ['train','validation','test']:
        selected=[row for row in rows if row['split']==split]
        if len(selected)<12: return False
        for axis in AXES:
            labels=[axis_label(axis,row['family_id']) for row in selected]
            if sum(label==0 for label in labels)<4 or sum(label==1 for label in labels)<4:
                return False
    return True
