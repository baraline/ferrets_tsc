"""Bounds of near-equal integer parts (the row and column splits of the Gram
and LSQR routes and the cut samplers)."""

import numpy as np


def _split(w, k):
    """k + 1 integer bounds splitting range(w) into k near-equal parts."""
    k = max(1, int(k))
    return (np.arange(k + 1, dtype=np.int64) * int(w)) // k
