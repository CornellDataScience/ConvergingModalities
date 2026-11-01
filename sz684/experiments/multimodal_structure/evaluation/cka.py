"""Linear CKA compares centered geometry, even when feature widths differ."""
import numpy as np


def linear_cka(left, right):
    """Return standard biased linear CKA for corresponding [N, D] rows."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right) or len(left) < 3:
        raise ValueError('CKA needs corresponding matrices with at least three rows')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Features must be finite')
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    # Gram matrices keep memory dependent on concept count, not feature width.
    a = left @ left.T
    b = right @ right.T
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-20:
        raise ValueError('CKA is undefined for constant representations')
    return float(np.clip(np.sum(a * b) / denominator, 0, 1))
