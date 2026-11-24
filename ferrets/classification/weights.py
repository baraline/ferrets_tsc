"""Class weights of the batch head: the weighted exact leave-one-out of the
Gram route and the weighted centred operator of the LSQR route.

``batch_class_weight="sqrt"`` weights each row of the ridge loss

    J(b, beta) = sum_i w_i ||y_i - b - beta^T x_i||^2 + alpha ||beta||^2

by its class's weight w_c = n_c ** -0.5 * n / sum_c' n_c' ** 0.5, which has
mean 1 over the fitted rows and is proportional to the streaming head's
(n / (K n_c)) ** 0.5. The intercept b is not penalised, the column means and
the target mean are the weighted ones (the decision of a row is
beta^T (x - mu_w) + ybar_w), and the prior shift reads the weighted ybar.
Exactly balanced classes give no weights: the unweighted branches run, bit
for bit."""

import numpy as np
from scipy.linalg import eigh as _scipy_eigh

from ..classification.common import _decide
from ..classification.flat_design import _CenteredOp
from ..kernels.flat import _fd_colmean_acc


BATCH_CLASS_WEIGHTS = ("sqrt",)


def _ridge_class_weights(y_idx, n_classes, rule):
    """The per-class row weights (n_classes,) of the weighted ridge, or None
    when unweighted: rule None, or exactly balanced classes (the weighted
    problem is the unweighted one). "sqrt": w_c = n_c ** -0.5 * n / sum_c'
    n_c' ** 0.5 over the classes present, so the weights have mean 1 over the
    rows; a class absent from y_idx weighs 0. A row takes its class's weight,
    w[y_idx]."""
    if rule is None:
        return None
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    present = counts > 0
    if counts[present].min() == counts[present].max():
        return None
    w = np.zeros(n_classes)
    w[present] = counts[present] ** -0.5
    w *= len(y_idx) / (counts[present] ** 0.5).sum()
    return w


def _gcv_w(K, xmu, mumu, Y, y_idx, alphas, w):
    """_gcv with row weights w (> 0): the exact weighted leave-one-out of the
    ridge with an unpenalised intercept, on the raw Gram K = X X^T
    (overwritten), xmu = X mu_w, mumu = mu_w . mu_w and w.

    With sw = sqrt(w), Xc = X - 1 mu_w^T, A = diag(sw) Xc and r = sw (Y -
    ybar_w): K_w = A A^T has sw as a zero-eigenvalue eigenvector (Xc^T w =
    0), and K_w + sw sw^T (the intercept column) is diagonalised with that
    eigenvector's weight zeroed. c = G_alpha r is the dual solution, c / g
    (g = diag(G_alpha)) the leave-one-out residual of the scaled problem; row
    i is predicted Y_i - (c_i / g_i) / sw_i by the fit without it. An alpha
    scores -mean((c / g)^2) = -mean(w eps^2); ties keep the earlier alpha.
    Returns _gcv's tuple, with the dual coefficients sw * c (for
    _dual_to_primal) and ybar_w."""
    n = K.shape[0]
    sw = np.sqrt(w)
    ybar = (w[:, None] * Y).sum(axis=0) / w.sum()
    Yr = sw[:, None] * (Y - ybar)
    K -= xmu[:, None]
    K -= xmu[None, :]
    K += mumu
    K *= sw[:, None]
    K *= sw[None, :]
    K += sw[:, None] * sw[None, :]
    eigvals, Q = _scipy_eigh(K, overwrite_a=True, check_finite=False,
                             driver="evd")
    QT_y = Q.T @ Yr
    icpt = int(np.argmax(np.abs((sw / np.linalg.norm(sw)) @ Q)))
    Q2 = Q * Q
    best, best_i, best_c, best_g = None, 0, None, None
    scores = np.empty(len(alphas))
    for a, alpha in enumerate(alphas):
        wd = 1.0 / (eigvals + alpha)
        wd[icpt] = 0.0
        c = Q @ (wd[:, None] * QT_y)
        g = Q2 @ wd
        score = -np.mean((c / g[:, None]) ** 2)
        scores[a] = score
        if best is None or score > best:
            best, best_i, best_c, best_g = score, a, c, g
    loo = Y - (best_c / best_g[:, None]) / sw[:, None]
    cv = float(np.count_nonzero(_decide(loo) == y_idx)) / n
    return best_i, scores, sw[:, None] * best_c, ybar, cv, loo


class _WeightedFlatOp:
    """diag(sqrt w) (X_S - 1 mu_w^T) on the rows S of a _FlatDesign,
    multi-column, with _CenteredOp's interface (shape, mu, matvec, rmatvec):
    mu is the weighted column mean (float32 sums, like _CenteredOp's) and the
    products of the raw X_S go through the uncentred products of a _CenteredOp
    (built on no rows, its rows set below), so the kernel dispatch (the column
    padding, the RMV_SCRATCH_MB rule, the spans) is the unweighted route's."""

    def __init__(self, D, rows, w_rows, n_threads):
        w_rows = np.asarray(w_rows, dtype=np.float64)
        core = _CenteredOp(D, np.empty(0, dtype=np.int64), n_threads)
        core.rows = np.ascontiguousarray(rows, dtype=np.int64)
        core.n_blocks = int(max(1, min(len(core.rows), 4 * n_threads)))
        core.shape = (len(core.rows), D.F)
        self.core = core
        self.D = D
        self.order = core.rows
        self.sw = np.sqrt(w_rows)
        self.shape = core.shape
        mu32 = np.zeros(D.F, dtype=np.float32)
        if len(core.rows) and len(core.rp):
            _fd_colmean_acc(D.ptr, D.ids, D.vals, D.off, core.rows,
                            (w_rows / w_rows.sum()).astype(np.float32),
                            core.rp, core.rj0, core.rj1, mu32)
        self.mu = mu32.astype(np.float64)

    def matvec(self, V):
        out = self.core.raw_matvec(V)           # X_S V
        out -= (self.mu @ V)[None, :]
        out *= self.sw[:, None]
        return out

    def rmatvec(self, U):
        Us = self.sw[:, None] * U
        out = self.core.raw_rmatvec(Us)         # X_S^T (sw U)
        out -= self.mu[:, None] * Us.sum(axis=0)[None, :]
        return out


def _check_batch_class_weight(v):
    if not (v is None or (isinstance(v, str) and v in BATCH_CLASS_WEIGHTS)):
        raise ValueError(f"batch_class_weight must be 'sqrt' or None, got "
                         f"{v!r}")
