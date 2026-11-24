"""Input collections and the windows parameter."""

import numba
import numpy as np
from aeon.utils.validation.collection import get_type

from ..constants import WORD_LEN
from ..kernels.finite import _count_nonfinite


def _check_X(self, X):
    """aeon's input check (``BaseCollectionEstimator._check_X``) without its
    variance floor. aeon refuses a series that varies with a standard
    deviation of at most 1e-7; here a window is flat when its standard
    deviation is at most EPS_STD and every finite series is keyed, so the
    estimators take this method in place of aeon's. Returns X's metadata."""
    get_type(X)                           # raises on an unknown collection
    metadata = self._get_X_metadata(X)
    problems = [name for key, name in (("missing_values", "missing values"),
                                       ("unequal_length",
                                        "unequal length series"))
                if metadata[key]]
    if problems:
        who = type(self).__name__
        raise ValueError(f"Data seen by instance of {who} has "
                         f"{' and '.join(problems)}, but {who} cannot handle "
                         "these characteristics. ")
    return metadata


def _as_collection(X):
    """(n, C, m) C-contiguous float64 (a 2-D input is one channel); a
    C-contiguous float64 input is not copied."""
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"X must be (n, m) or (n, C, m), got shape {X.shape}")
    if X.dtype.kind not in "fiub":
        raise ValueError(f"X must be numeric, got dtype {X.dtype}")
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.size:
        nb = int(min(X.shape[0], 4 * max(1, numba.get_num_threads())))
        if _count_nonfinite(X, nb) > 0:
            raise ValueError("X contains NaN or infinite values")
    return X


def _is_int(v):
    """True for an int or numpy integer, False for a bool."""
    return isinstance(v, (int, np.integer)) and not isinstance(
        v, (bool, np.bool_))


def _check_windows(windows):
    """windows as a tuple of ints, each a multiple of WORD_LEN (at least
    WORD_LEN); raises ValueError otherwise."""
    try:
        ws = tuple(int(w) for w in windows)
    except TypeError:
        raise ValueError("windows must be a sequence of ints, got "
                         f"{windows!r}")
    if not ws:
        raise ValueError("windows must not be empty")
    for w, raw in zip(ws, windows):
        if w != raw or w < WORD_LEN or w % WORD_LEN:
            raise ValueError(f"every window must be an int multiple of "
                             f"{WORD_LEN} (>= {WORD_LEN}), got {raw!r}")
    if len(set(ws)) != len(ws):
        raise ValueError(f"duplicate windows in {ws}")
    return ws
