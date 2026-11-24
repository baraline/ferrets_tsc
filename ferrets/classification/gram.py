"""The Gram / GCV ridge route (n_train <= DUAL_MAX_N)."""

import time

import numpy as np
from scipy.linalg import eigh as _scipy_eigh, get_blas_funcs

from .. import config
from ..classification.common import _decide
from ..helpers.split import _split
from ..kernels.flat import _fd_colmean, _fd_colmean_acc
from ..kernels.gram import (_densify_rows, _dual_to_primal, _mirror_upper,
                            _rows_dot_acc)
from ..classification.weights import _gcv_w
from ..transform.design import _PieceBufs


def _gcv(K, xmu, mumu, Y, y_idx, alphas):
    """RidgeClassifierCV's eigen/GCV selection on the raw Gram K = X X^T
    (overwritten): the sparse centring formula plus 1 1^T for the intercept,
    eigh, the intercept eigenvector zeroed, the LOO squared error of every
    alpha, ties keep the earlier alpha. Returns (alpha index, scores, dual
    coefficients C (n, K'), y_offset, LOO accuracy, LOO decisions (n, K') of
    the chosen alpha)."""
    n = K.shape[0]
    y_offset = Y.mean(axis=0)
    Yc = Y - y_offset
    K -= xmu[:, None]
    K -= xmu[None, :]
    K += mumu + 1.0
    eigvals, Q = _scipy_eigh(K, overwrite_a=True, check_finite=False,
                             driver="evd")
    QT_y = Q.T @ Yc
    ones = np.full(n, 1.0 / np.sqrt(n))
    icpt = int(np.argmax(np.abs(ones @ Q)))
    Q2 = Q * Q
    best, best_i, best_c, best_g = None, 0, None, None
    scores = np.empty(len(alphas))
    for a, alpha in enumerate(alphas):
        w = 1.0 / (eigvals + alpha)
        w[icpt] = 0.0
        c = Q @ (w[:, None] * QT_y)
        g = Q2 @ w
        score = -np.mean((c / g[:, None]) ** 2)
        scores[a] = score
        if best is None or score > best:
            best, best_i, best_c, best_g = score, a, c, g
    loo = Y - best_c / best_g[:, None]
    cv = float(np.count_nonzero(_decide(loo) == y_idx)) / n
    return best_i, scores, best_c, y_offset, cv, loo


class GramMixin:
    """The Gram / GCV ridge route.

    Sets: alpha_, alpha_scores_, coef_, cv_score_, intercept_, solver_.
    Reads: _n_jobs, batch_class_weight_."""

    def _gram_packs(self, n, off):
        """The Gram's column pieces (block, c0, c1, slab offset) packed into
        slabs, and the widest slab, for n rows and the blocks' first columns
        ``off`` (the design's offsets). A block wider than GRAM_SLAB_MB of
        slab spans several pieces. Narrow blocks share a slab (one BLAS call
        for several), but a slab grows past the widest single block only
        while it stays under GRAM_PACK_MB, so a slab is never larger than the
        larger of the widest block's and GRAM_PACK_MB."""
        widths = off[1:] - off[:-1]
        s_cols = max(1, int(config.GRAM_SLAB_MB * 1e6 // (8 * max(1, n))))
        widest = min(s_cols, int(widths.max()) if len(widths) else 1)
        pack_cols = min(s_cols, max(widest, int(config.GRAM_PACK_MB * 1e6
                                                // (8 * max(1, n)))))
        packs, cur, fill = [], [], 0
        for p in range(len(widths)):
            w = int(widths[p])
            for c0 in range(0, w, s_cols):
                c1 = min(w, c0 + s_cols)
                if fill + (c1 - c0) > pack_cols and cur:
                    packs.append((cur, fill))
                    cur, fill = [], 0
                cur.append((p, c0, c1, fill))
                fill += c1 - c0
        if cur:
            packs.append((cur, fill))
        return packs, max((f for _, f in packs), default=1)

    def _fit_gram(self, y_idx, Y, alphas, design, stats):
        """Accumulates K = X X^T slab by slab from the training design,
        selects alpha by the exact leave-one-out, maps the dual coefficients
        back to coef_ (F, K') and sets intercept_; timings go to stats."""
        tt = time.perf_counter
        t0 = tt()
        n = design.n
        # batch_class_weight_ None is the unweighted ridge; otherwise the
        # column means and the leave-one-out are weighted
        wrow = (None if self.batch_class_weight_ is None
                else self.batch_class_weight_[y_idx])
        K_mat, mu, xmu, packs, width_max = self._gram_matrix(design, wrow)
        stats["t_gram"] = tt() - t0
        stats["n_slabs"] = len(packs)
        stats["slab_bytes"] = int(8 * n * max(1, width_max))
        t1 = tt()
        if wrow is None:
            a_i, scores, Cd, y_offset, cv, loo = _gcv(
                K_mat, xmu, mu @ mu, Y, y_idx, np.asarray(alphas))
        else:
            a_i, scores, Cd, y_offset, cv, loo = _gcv_w(
                K_mat, xmu, mu @ mu, Y, y_idx, np.asarray(alphas), wrow)
        del K_mat
        stats["t_solve"] = tt() - t1
        t2 = tt()
        self.coef_, dot = self._primal_coef(design, Cd, mu)
        self.intercept_ = y_offset - dot
        self.alpha_ = float(alphas[a_i])
        self.alpha_scores_ = tuple(float(s) for s in scores)
        self.cv_score_ = float(cv)
        self.solver_ = "gcv"
        stats["t_coef"] = tt() - t2
        self._apply_prior(loo, y_idx, y_offset, stats)

    def _gram_matrix(self, design, wrow):
        """The raw Gram K = X X^T (n, n, Fortran order, both triangles) of the
        design's rows, the column means mu (F,) (weighted by wrow, the row
        weights, unless None), xmu = X mu (n,), the slab packs and the
        column count of the widest slab."""
        n = design.n
        F = design.n_features
        off = design.offsets
        rw32 = None if wrow is None else (wrow / wrow.sum()).astype(
            np.float32)
        K_mat = np.zeros((n, n), order="F")
        # one triangle of every slab's product, whatever n
        syrk = get_blas_funcs("syrk", (K_mat,))
        mu = np.empty(F)
        xmu = np.zeros(n)
        scale32 = np.float32(1.0 / n)
        packs, width_max = self._gram_packs(n, off)
        slab = np.empty(n * max(1, width_max))
        n_rng = 4 * self._n_jobs
        off0 = np.zeros(1, dtype=np.int64)
        all_rows = np.arange(n, dtype=np.int64)
        # a block's pieces are consecutive (across packs too): a row cursor
        # per block decodes one piece's CSR at a time from the encoded
        # design, and the unchanged kernels run on it, a piece's column means
        # before its row dots, pieces in ascending column order
        bufs = _PieceBufs(design.id_dtype)
        held_p, cur, mu_p, mu32 = -1, None, None, None
        ptr = ids = vals = None
        for pieces, width in packs:
            Tm = slab[:n * width].reshape((n, width))
            for (p, c0, c1, o0) in pieces:
                if p != held_p:
                    cur = design.cursor(p, bufs)
                    held_p = p
                    mu32 = np.empty(int(off[p + 1] - off[p]),
                                    dtype=np.float32)
                    mu_p = mu[off[p]:off[p + 1]]
                ptr, ids, vals = cur.piece(c0, c1)
                bnd = c0 + _split(c1 - c0, max(1, min(n_rng, c1 - c0)))
                if wrow is None:
                    _fd_colmean(ptr.reshape(1, -1), ids, vals, off0, all_rows,
                                np.zeros(len(bnd) - 1, dtype=np.int64),
                                bnd[:-1], bnd[1:], scale32, mu32)
                else:
                    mu32[c0:c1] = 0.0
                    _fd_colmean_acc(ptr.reshape(1, -1), ids, vals, off0,
                                    all_rows, rw32,
                                    np.zeros(len(bnd) - 1, dtype=np.int64),
                                    bnd[:-1], bnd[1:], mu32)
                mu_p[c0:c1] = mu32[c0:c1]
                _rows_dot_acc(ptr, ids, vals, mu_p, xmu)
                _densify_rows(ptr, ids, vals, np.int64(c0), np.int64(c1), Tm,
                              np.int64(o0))
            out = syrk(1.0, Tm.T, beta=1.0, c=K_mat, trans=1, lower=0,
                       overwrite_c=1)
            if not np.shares_memory(out, K_mat):
                K_mat = np.asfortranarray(out)
        cur = ptr = ids = vals = bufs = None
        _mirror_upper(K_mat)
        return K_mat, mu, xmu, packs, width_max

    def _primal_coef(self, design, Cd, mu):
        """coef (F, K') float32 from the dual coefficients Cd (n, K'), block
        by block through the design's cursors, and dot = mu . coef (K',)."""
        n = design.n
        F = design.n_features
        off = design.offsets
        n_rng = 4 * self._n_jobs
        Kp = Cd.shape[1]
        coef = np.empty((F, Kp), dtype=np.float32)
        dot = np.zeros(Kp)
        Cd = np.ascontiguousarray(Cd)
        # the pieces keep the slab pass's width, so the piece buffers stay
        # within the same bound; any partition gives the same sums (a
        # column's rows are summed ascending, whatever the split)
        s_cols = max(1, int(config.GRAM_SLAB_MB * 1e6 // (8 * max(1, n))))
        bufs = _PieceBufs(design.id_dtype)
        for p in range(design.n_blocks):
            w = int(off[p + 1] - off[p])
            if w == 0:
                continue
            cur = design.cursor(p, bufs)
            cp = np.empty((w, Kp))
            for c0 in range(0, w, s_cols):
                c1 = min(w, c0 + s_cols)
                ptr, ids, vals = cur.piece(c0, c1)
                bnd = c0 + _split(c1 - c0, max(1, min(n_rng, c1 - c0)))
                _dual_to_primal(ptr, ids, vals, Cd, bnd, cp)
            coef[off[p]:off[p + 1]] = cp
            dot += mu[off[p]:off[p + 1]] @ cp
            del cp
        return coef, dot
