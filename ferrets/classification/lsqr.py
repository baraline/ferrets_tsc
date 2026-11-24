"""The lockstep multi-damping LSQR ridge route."""

import math
import time

import numpy as np
from sklearn.model_selection import train_test_split

from .. import config
from ..classification.common import _decide
from ..classification.flat_design import (_U16_BITS, _CenteredOp,
                                          _FlatDesign, _FlatRaw, _sub_widths)
from ..kernels.flat import _fd_matvec
from ..classification.weights import _WeightedFlatOp
from ..kernels.krylov import _gk_orth, _gk_store


def _bidiag(a, b, kk, beta1):
    """The projected problem of k = kk Golub-Kahan steps: the (kk + 1) x kk
    lower-bidiagonal B_k (alphas a on the diagonal, betas b below it) and
    beta1 e1."""
    Bm = np.zeros((kk + 1, kk))
    Bm[np.arange(kk), np.arange(kk)] = a[:kk]
    Bm[np.arange(1, kk + 1), np.arange(kk)] = b[:kk]
    e1 = np.zeros(kk + 1)
    e1[0] = beta1
    return Bm, e1


def _damped_lstsq(Bm, e1, lam):
    """argmin_y ||Bm y - e1||^2 + lam ||y||^2, as one stacked least squares."""
    kk = Bm.shape[1]
    M = np.vstack([Bm, np.sqrt(lam) * np.eye(kk)])
    return np.linalg.lstsq(M, np.concatenate([e1, np.zeros(kk)]),
                           rcond=None)[0]


def _solve_and_test(a, b, kk, beta1, al, damps_sq, tol, ys):
    """One column after k = kk Golub-Kahan steps (alphas a, betas b, the next
    alpha al, first beta beta1): the projected solution of every damping into
    ys, and whether all of them pass LSQR's two stopping rules."""
    Bm, e1 = _bidiag(a, b, kk, beta1)
    sq_ab = (np.sum(np.square(np.asarray(a[:kk])))
             + np.sum(np.square(np.asarray(b[:kk]))))
    all_done = True
    for i, lam in enumerate(damps_sq):
        y = _damped_lstsq(Bm, e1, lam)
        ys[i] = y
        res = Bm @ y - e1
        r1norm = np.linalg.norm(res)
        rnorm = np.sqrt(res @ res + lam * (y @ y))
        arnorm = np.sqrt(np.sum((Bm.T @ res + lam * y) ** 2)
                         + (al * res[-1]) ** 2)
        anorm = np.sqrt(sq_ab + al * al + kk * lam)
        xnorm = np.linalg.norm(y)
        test1 = r1norm <= tol * beta1 + tol * anorm * xnorm
        test2 = (rnorm == 0) or (arnorm / (anorm * rnorm) <= tol)
        if not (test1 or test2):
            all_done = False
    return all_done


def _lsqr_group(op, B, damps_sq, tol, max_iter, coef_out):
    """Damped least squares min ||A x - b||^2 + lam ||x||^2 for several lam
    and the G columns b of B at once, into coef_out (n_damps, G, F).

    Each column runs one Golub-Kahan bidiagonalisation with a
    re-orthogonalised Krylov basis; every lam is solved on the projected
    (k+1) x k problem and tested with LSQR's two stopping rules
    (atol = btol = tol), and a column stops when all its lams pass. The
    columns advance in lockstep, so every matvec / rmatvec is one pass over
    the design for all active columns. The per-column vector work is one
    fused parallel kernel per column (_gk_orth / _gk_store). A basis starts
    at 16 rows and doubles. An exact breakdown (a new beta of 0: the Krylov
    space is invariant) ends the column with the projected solve at the
    current k, which is the exact solution in that space."""
    # Per column g: Vv right vector, U left vector, alpha / beta its current
    # Golub-Kahan scalars; Vb[g] the basis, a_gk / b_gk the alphas and betas
    # so far, ys[g] the projected solutions, act the columns still running.
    F = op.shape[1]
    G = B.shape[1]
    nd = len(damps_sq)
    coef_out[...] = 0.0
    beta1 = np.sqrt((B * B).sum(axis=0))
    U = np.zeros_like(B)
    ok = beta1 > 0
    U[:, ok] = B[:, ok] / beta1[ok]
    Vv = np.ascontiguousarray(op.rmatvec(U))
    alpha = np.sqrt((Vv * Vv).sum(axis=0))
    act = [g for g in range(G) if beta1[g] > 0 and alpha[g] > 0]
    cap0 = max(1, min(int(max_iter), 16))
    Vb, a_gk, b_gk, ys = {}, {}, {}, {}
    vn = np.empty(F)
    for g in act:
        Vv[:, g] /= alpha[g]
        Vb[g] = np.empty((cap0, F))
        Vb[g][0] = Vv[:, g]
        a_gk[g] = [alpha[g]]
        b_gk[g] = []
        ys[g] = [None] * nd
    while act:
        idx = np.asarray(act, dtype=np.int64)
        Uw = op.matvec(np.take(Vv, idx, axis=1)) \
            - alpha[idx][None, :] * U[:, idx]
        beta = np.sqrt((Uw * Uw).sum(axis=0))
        done, live = [], []
        for gi, g in enumerate(act):
            if beta[gi] == 0:
                kk = len(a_gk[g])
                Bm, e1 = _bidiag(a_gk[g], b_gk[g] + [0.0], kk, beta1[g])
                ys[g] = [_damped_lstsq(Bm, e1, lam) for lam in damps_sq]
                done.append(g)
            else:
                U[:, g] = Uw[:, gi] / beta[gi]
                b_gk[g].append(beta[gi])
                live.append(g)
        if live:
            li = np.asarray(live, dtype=np.int64)
            Vw = op.rmatvec(np.take(U, li, axis=1))
            for c_, g in enumerate(live):
                kk = len(a_gk[g])
                V = Vb[g]
                al = np.float64(_gk_orth(Vw, np.int64(c_), Vv, np.int64(g),
                                         np.float64(b_gk[g][-1]), V,
                                         np.int64(kk), vn))
                all_done = _solve_and_test(a_gk[g], b_gk[g], kk, beta1[g], al,
                                           damps_sq, tol, ys[g])
                if all_done or al == 0 or kk >= max_iter:
                    done.append(g)
                else:
                    if kk >= V.shape[0]:
                        grown = np.empty((min(int(max_iter),
                                              2 * V.shape[0]), F))
                        grown[:kk] = V[:kk]
                        Vb[g] = V = grown
                    _gk_store(vn, al, V[kk], Vv, np.int64(g))
                    alpha[g] = al
                    a_gk[g].append(al)
        for g in done:
            yl = [y for y in ys[g] if y is not None]
            if yl:
                # every damping's y has the final k: one product for all
                Y_ = np.stack(yl)
                coef_out[:len(yl), g] = Y_ @ Vb[g][:Y_.shape[1]]
            Vb.pop(g, None)
        dset = set(done)
        act = [g for g in act if g not in dset]
    return coef_out


def _lsqr_solve(op, Y, alphas, group, tol=None):
    """coef (n_alphas, K', F) and intercepts (n_alphas, K') of the centred
    ridge for targets Y, the classes solved in groups of ``group``. ``tol``
    is the LSQR tolerance (config.RIDGE_TOL when None)."""
    tol = config.RIDGE_TOL if tol is None else tol
    y_offset = Y.mean(axis=0)
    Yc = Y - y_offset
    K = Yc.shape[1]
    damps = np.asarray(alphas, dtype=np.float64)
    coef = np.empty((len(alphas), K, op.shape[1]))
    for g0 in range(0, K, group):
        g1 = min(K, g0 + group)
        _lsqr_group(op, np.ascontiguousarray(Yc[:, g0:g1]), damps,
                    tol, config.RIDGE_MAX_ITER, coef[:, g0:g1, :])
    icpt = y_offset[None, :] - coef @ op.mu
    return coef, icpt


def _lsqr_holdout(D, tr, va, Y, alphas, group, n_threads, tol=None):
    """Hold-out decisions (n_alphas, len(va), K') of the ridge fitted on the
    rows tr for every alpha, class group by class group: a group's
    coefficients are scored on the validation rows and dropped, so the
    hold-out never holds every alpha's (K', F) coefficients at once. They
    live in one (F, n_alphas, G) array (_lsqr_group writes them through its
    transposed view), so the group is one validation pass of n_alphas x G
    columns (its flat (F, n_alphas G) view, no copy). The values are
    _lsqr_solve's: _fd_matvec's column sums do not depend on the other
    columns, and the intercept products run on one (K', F) buffer holding the
    group's rows (BLAS's gemv result for a row depends on the matrix's shape,
    not on the other rows' values). ``tol`` is the LSQR tolerance
    (config.RIDGE_TOL when None)."""
    tol = config.RIDGE_TOL if tol is None else tol
    op = _CenteredOp(D, tr, n_threads)
    Ytr = Y[tr]
    y_offset = Ytr.mean(axis=0)
    Yc = Ytr - y_offset
    K = Yc.shape[1]
    nA = len(alphas)
    damps = np.asarray(alphas, dtype=np.float64)
    nb = np.int64(max(1, min(len(va), 4 * n_threads)))
    dec = np.empty((nA, len(va), K))
    buf = np.zeros((K, D.F))
    for g0 in range(0, K, group):
        g1 = min(K, g0 + group)
        G = g1 - g0
        cf = np.empty((D.F, nA, G))
        _lsqr_group(op, np.ascontiguousarray(Yc[:, g0:g1]), damps,
                    tol, config.RIDGE_MAX_ITER, cf.transpose(1, 2, 0))
        d = np.empty((len(va), nA * G))
        _fd_matvec(D.ptr, D.ids, D.vals, D.off, va, cf.reshape(D.F, nA * G),
                   d, nb)
        for a in range(nA):
            buf[g0:g1] = cf[:, a, :].T
            dec[a, :, g0:g1] = d[:, a * G:(a + 1) * G] + (
                y_offset[g0:g1] - (buf @ op.mu)[g0:g1])[None, :]
        del cf, d
    return dec


def _lsqr_holdout_ops(op, raw_va, Ytr, w_tr, alphas, group, F):
    """_lsqr_holdout over operators, with row weights w_tr: the targets and
    the centring are weighted. The decisions (n_alphas, n_va, K') of the
    ridge fitted on op's rows (a _WeightedFlatOp, in the order of Ytr; w_tr
    its row weights, mean 1, or None: the unweighted centred ridge) for every
    alpha, class group by class group. raw_va holds the validation
    rows (``order``, and matvec(V) = X_va V). The weighted fit is the
    centred problem diag(sw)(X - 1 mu_w^T) b = sw (Y - ybar_w); a validation
    row is decided X_va b + (ybar_w - mu_w . b), the fit's own prediction."""
    if w_tr is None:
        y_offset = Ytr.mean(axis=0)
        rhs = Ytr - y_offset
    else:
        y_offset = (w_tr[:, None] * Ytr).sum(axis=0) / w_tr.sum()
        rhs = np.sqrt(w_tr)[:, None] * (Ytr - y_offset)
    K = rhs.shape[1]
    nA = len(alphas)
    damps = np.asarray(alphas, dtype=np.float64)
    dec = np.empty((nA, len(raw_va.order), K))
    buf = np.zeros((K, F))
    for g0 in range(0, K, group):
        g1 = min(K, g0 + group)
        G = g1 - g0
        cf = np.empty((F, nA, G))
        _lsqr_group(op, np.ascontiguousarray(rhs[:, g0:g1]), damps,
                    config.RIDGE_TOL, config.RIDGE_MAX_ITER,
                    cf.transpose(1, 2, 0))
        d = raw_va.matvec(cf.reshape(F, nA * G))
        for a in range(nA):
            buf[g0:g1] = cf[:, a, :].T
            dec[a, :, g0:g1] = d[:, a * G:(a + 1) * G] + (
                y_offset[g0:g1] - (buf @ op.mu)[g0:g1])[None, :]
        del cf, d
    return dec


def _lsqr_solve_ops(op, Y, w, alpha, group):
    """_lsqr_solve over operators for one alpha, with row weights w: the
    targets and the centring are weighted. Returns (coef (K', F),
    intercept (K',), target mean (K',)) of the ridge on the rows of op (a
    _WeightedFlatOp whose row weights are w, in the order of Y; w None:
    unweighted, the mean the plain one)."""
    if w is None:
        y_offset = Y.mean(axis=0)
        rhs = Y - y_offset
    else:
        y_offset = (w[:, None] * Y).sum(axis=0) / w.sum()
        rhs = np.sqrt(w)[:, None] * (Y - y_offset)
    K = rhs.shape[1]
    coef = np.empty((1, K, op.shape[1]))
    for g0 in range(0, K, group):
        g1 = min(K, g0 + group)
        _lsqr_group(op, np.ascontiguousarray(rhs[:, g0:g1]),
                    np.asarray([alpha], dtype=np.float64), config.RIDGE_TOL,
                    config.RIDGE_MAX_ITER, coef[:, g0:g1, :])
    return coef[0], y_offset - coef[0] @ op.mu, y_offset


def _group_size(K, F):
    """Class columns solved in lockstep: at most LSQR_GROUP, and few enough
    for their Krylov bases (LSQR_K_GUESS rows of F floats each) to fit
    LSQR_STATE_MB."""
    return int(max(1, min(K, config.LSQR_GROUP, config.LSQR_STATE_MB * 1e6
                          // max(1, F * 8 * config.LSQR_K_GUESS))))


def _holdout_split(y_idx, seed):
    """The 25% hold-out (tr, va) of the LSQR route's alpha choice:
    stratified when every class has two rows and the hold-out has a row per
    class."""
    n = len(y_idx)
    counts = np.bincount(y_idx)
    n_test = math.ceil(config.HOLDOUT * n)
    strat = y_idx if (counts.min() >= 2 and n_test >= len(counts)) else None
    return train_test_split(np.arange(n), test_size=config.HOLDOUT,
                            random_state=seed, stratify=strat)


class LsqrMixin:
    """The LSQR ridge route.

    Sets: alpha_, alpha_scores_, coef_, cv_score_, intercept_, solver_.
    Reads: _n_jobs, batch_class_weight_, transformer_."""

    def _build_flat_design(self, design):
        """The LSQR flat design of the batch blocks (also the ablation's
        ridge-CV head). With config.FLAT_BAND_BITS <= _U16_BITS every kept
        block p wider than 1 << FLAT_BAND_BITS columns is cut into
        consecutive sub-blocks of that width, ids local to the sub-block and
        uint16 whatever the transform's id type (8 bytes per nonzero become 6
        when the transform's ids are int32); above _U16_BITS each block p
        stays whole, in the transform's id type. Both layouts hold the same
        entries in the same per-row order. Blocks wider than one band are
        decoded by a count pass and a banded decode; the others by
        design.csr (_dec_batch)."""
        kptrs = design.kptrs
        n = design.n
        bits = int(config.FLAT_BAND_BITS)
        widths = design.offsets[1:] - design.offsets[:-1]
        sub_w, nbs = _sub_widths(widths, bits)
        D = _FlatDesign(n, sub_w, design.nnz,
                        design.id_dtype if bits > _U16_BITS else np.uint16,
                        group=np.repeat(np.arange(len(nbs)), nbs))
        at = 0
        s = 0
        for p in range(design.n_blocks):
            kptr = kptrs[p]
            nk = int(kptr[-1])
            nb = int(nbs[p])
            if nb == 1:
                D.ptr[s] = kptr + at
                if nk:
                    design.csr(p, ids_buf=D.ids[at:at + nk],
                               vals_buf=D.vals[at:at + nk])
            else:
                cnt = design.band_counts(p, bits, nb)
                cs = np.cumsum(cnt, axis=0)
                if int(cs[-1].sum()) != nk:
                    raise RuntimeError("band counts disagree with the kept "
                                       "counts")
                start = at + np.concatenate(([0], np.cumsum(cs[-1])[:-1]))
                D.ptr[s:s + nb, 0] = start
                D.ptr[s:s + nb, 1:] = start[:, None] + cs.T
                design.decode_bands(
                    p, bits, np.ascontiguousarray(start[None, :] + cs - cnt),
                    D.ids, D.vals)
            design.release(p)
            at += nk
            s += nb
        return D

    def _fit_lsqr(self, y_idx, Y, alphas, design, stats):
        """Chooses alpha on a hold-out (when there are several), refits on all
        rows by LSQR into coef_ (F, K') and intercept_; timings go to stats."""
        tt = time.perf_counter
        t0 = tt()
        n = design.n
        D = self._build_flat_design(design)
        stats["t_flat"] = tt() - t0
        t1 = tt()
        group = _group_size(Y.shape[1], D.F)
        tr, va = _holdout_split(y_idx, self.transformer_.seed_)
        # hold-out hits per alpha, integer counts: the first most accurate
        # alpha wins, accuracies reported as Python floats.
        # batch_class_weight_ None is the unweighted ridge; otherwise the
        # hold-out fits on tr with the full set's class weights renormalised
        # to mean 1 over tr, so a balanced training set stays unweighted in
        # every fit
        wc = self.batch_class_weight_
        if len(alphas) > 1:
            va = np.ascontiguousarray(va, dtype=np.int64)
            if wc is None:
                dec = _lsqr_holdout(D, tr, va, Y, alphas, group, self._n_jobs)
            else:
                w_tr = wc[y_idx[tr]]
                w_tr = w_tr / w_tr.mean()
                dec = _lsqr_holdout_ops(
                    _WeightedFlatOp(D, tr, w_tr, self._n_jobs),
                    _FlatRaw(D, va, self._n_jobs), Y[tr], w_tr, alphas,
                    group, D.F)
            hits = [int(np.count_nonzero(_decide(d) == y_idx[va]))
                    for d in dec]
            best = int(np.argmax(hits))
            self.cv_score_ = hits[best] / len(va)
            self.alpha_scores_ = tuple(h / len(va) for h in hits)
            oos = (dec[best], y_idx[va])
        else:
            best = 0
            self.cv_score_ = None
            self.alpha_scores_ = None
            oos = (None, None)
        stats["t_holdout"] = tt() - t1
        t2 = tt()
        if wc is None:
            op = _CenteredOp(D, np.arange(n), self._n_jobs)
            coef, icpt = _lsqr_solve(op, Y, (alphas[best],), group)
            coef0, icpt0, ybar = coef[0], icpt[0], Y.mean(axis=0)
        else:
            w_all = wc[y_idx]
            op = _WeightedFlatOp(D, np.arange(n), w_all, self._n_jobs)
            coef0, icpt0, ybar = _lsqr_solve_ops(op, Y, w_all, alphas[best],
                                                 group)
        self.coef_ = np.ascontiguousarray(coef0.T, dtype=np.float32)
        self.intercept_ = np.asarray(icpt0, dtype=np.float64)
        self.alpha_ = float(alphas[best])
        self.solver_ = "lsqr"
        stats["t_solve"] = tt() - t2
        stats["lsqr_group"] = group
        self._apply_prior(oos[0], oos[1], ybar, stats)
