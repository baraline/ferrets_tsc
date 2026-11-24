"""Targets, decisions and the equal-weight score."""

import numpy as np


def _targets(y_idx, n_classes):
    """{-1, +1} one-vs-rest targets; two classes give the single column of
    the positive class (classes_[1])."""
    n = y_idx.shape[0]
    if n_classes == 2:
        Y = np.full((n, 1), -1.0)
        Y[y_idx == 1, 0] = 1.0
        return Y
    Y = np.full((n, n_classes), -1.0)
    Y[np.arange(n), y_idx] = 1.0
    return Y


def _decide(d):
    """Class index of each decision row (one column: positive = index 1)."""
    if d.shape[1] == 1:
        return (d[:, 0] > 0).astype(np.int64)
    return np.argmax(d, axis=1)


def _eq_score(y, p, n_classes):
    """(accuracy + balanced accuracy + macro F1) / 3, sklearn's
    definitions: balanced accuracy averages the recall of the classes present
    in y; macro F1 averages 2TP / (|true| + |predicted|) over all classes, 0
    where that is 0 / 0."""
    acc = float(np.mean(p == y))
    rec, f1 = [], []
    for k in range(n_classes):
        t = y == k
        q = p == k
        tp = float(np.count_nonzero(t & q))
        nt = int(np.count_nonzero(t))
        if nt:
            rec.append(tp / nt)
        den = nt + int(np.count_nonzero(q))
        f1.append(2.0 * tp / den if den else 0.0)
    return (acc + float(np.mean(rec)) + float(np.mean(f1))) / 3.0
