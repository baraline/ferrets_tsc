"""The ablation's alternative batch heads: logistic SGD and k-fold ridge
CV."""

import time

import numba
import numpy as np
from scipy import sparse

from ..classification.common import _decide, _targets
from ..classification.flat_design import _CenteredOp
from ..classification.lsqr import (_group_size, _lsqr_holdout,
                                   _lsqr_solve)


CV_RIDGE_TOL = 1e-4        # LSQR tolerance of the ridge-CV head
LOGISTIC = dict(loss="log_loss", penalty="l2", alpha=1e-5, average=True,
                max_iter=3, tol=None, n_iter_no_change=5)


class AblationHeadsMixin:
    """The logistic and ridge-CV heads of FERRETSAblation. It comes first in
    the bases, so its _fit_batch overrides the classifier's, which its super()
    reaches.

    Sets: alpha_, alpha_scores_, coef_, cv_score_, intercept_, n_epochs_,
    prior_lambda_, prior_scores_, sgd_n_jobs_, solver_.
    Reads: _n_jobs, transformer_."""

    def _batch_keep_all(self, n):
        """The logistic head decodes the blocks one at a time, the ridge-CV
        head flattens them all, the default head is the classifier's."""
        if self.head == "logistic":
            return False
        if self.ridge_cv is not None:
            return True
        return super()._batch_keep_all(n)

    def _fit_batch(self, design, y_idx, K, alphas, stats):
        if self.head == "logistic":
            return self._fit_logistic(design, y_idx, K, stats)
        if self.ridge_cv is not None:
            return self._fit_ridge_cv(design, y_idx, K, alphas, stats)
        return super()._fit_batch(design, y_idx, K, alphas, stats)

    def _design_csr(self, design):
        """The normalised batch design as one (n, F) float32 CSR with int32
        indices (sklearn's SGD takes nothing wider), assembled from the
        pseudo-channel blocks, which it frees as it copies them. Each row's
        columns ascend."""
        kptrs = design.kptrs
        n = design.n
        F = design.n_features
        off = design.offsets
        cnt = np.zeros(n, dtype=np.int64)
        for kp in kptrs:
            cnt += np.diff(kp)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(cnt, out=indptr[1:])
        total = int(indptr[-1])
        lim = np.iinfo(np.int32).max
        if total > lim or F > lim:
            raise ValueError(f"the batch design ({total} nonzeros, {F} "
                             "columns) exceeds sklearn's 32-bit indices")
        indices = np.empty(total, dtype=np.int32)
        data = np.empty(total, dtype=np.float32)
        pos = indptr[:-1].copy()
        for p in range(len(kptrs)):
            if int(kptrs[p][-1]) == 0:
                continue
            kptr, ids, vals = design.csr(p)
            c = np.diff(kptr)
            dest = np.repeat(pos - kptr[:-1], c) + np.arange(int(kptr[-1]))
            indices[dest] = ids.astype(np.int64) + int(off[p])
            data[dest] = vals
            pos += c
            design.release(p)
            del kptr, ids, vals, dest
        return sparse.csr_matrix((data, indices, indptr.astype(np.int32)),
                                 shape=(n, F))

    def _fit_logistic(self, design, y_idx, K, stats):
        """Logistic SGD (LOGISTIC) on the normalised design; the prior shift
        does not apply (lambda 0)."""
        from sklearn.linear_model import SGDClassifier
        tt = time.perf_counter
        t1 = tt()
        D = self._design_csr(design)
        stats["t_assemble"] = tt() - t1
        t2 = tt()
        n_jobs = int(numba.get_num_threads())
        clf = SGDClassifier(random_state=self.transformer_.seed_,
                            n_jobs=n_jobs, **LOGISTIC)
        clf.fit(D, y_idx)
        stats["t_solve"] = tt() - t2
        stats["recomputed"] = int(design.recomputed)
        self.coef_ = np.ascontiguousarray(
            np.asarray(clf.coef_, dtype=np.float32).T)
        self.intercept_ = np.asarray(clf.intercept_, dtype=np.float64).copy()
        self.alpha_ = None
        self.alpha_scores_ = None
        self.cv_score_ = None
        self.solver_ = "logistic_sgd"
        self.n_epochs_ = int(clf.n_iter_)
        self.sgd_n_jobs_ = n_jobs
        self.prior_lambda_ = 0.0
        self.prior_scores_ = None
        stats["prior_lambda"] = 0.0

    def _fit_ridge_cv(self, design, y_idx, K, alphas, stats):
        """LSQR ridge with alpha chosen by stratified k-fold CV (k =
        ridge_cv, capped by the smallest class count but at least 2),
        refitted on every row. The best alpha's out-of-fold decisions feed
        the prior shift. A single alpha skips the CV and the shift."""
        from sklearn.model_selection import StratifiedKFold
        tt = time.perf_counter
        n = design.n
        t1 = tt()
        D = self._build_flat_design(design)
        stats["t_flat"] = tt() - t1
        t2 = tt()
        Y = _targets(y_idx, K)
        Kp = Y.shape[1]
        group = _group_size(Kp, D.F)
        k = int(self.ridge_cv)
        if len(alphas) > 1:
            n_splits = int(min(k, max(2, np.bincount(y_idx).min())))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=self.transformer_.seed_)
            accs = np.zeros(len(alphas))
            folds = []
            for tr, va in skf.split(np.zeros((n, 1)), y_idx):
                tr = np.ascontiguousarray(tr, dtype=np.int64)
                va = np.ascontiguousarray(va, dtype=np.int64)
                dec = _lsqr_holdout(D, tr, va, Y, alphas, group,
                                    self._n_jobs, CV_RIDGE_TOL)
                folds.append((va, dec))
                for a in range(len(alphas)):
                    accs[a] += float(np.mean(_decide(dec[a])
                                             == y_idx[va]))
            best = int(np.argmax(accs))     # the first most accurate alpha
            self.cv_score_ = float(accs[best] / n_splits)
            self.alpha_scores_ = tuple(float(a / n_splits) for a in accs)
            stats["cv_folds"] = n_splits
            oos = (np.concatenate([d[best] for _, d in folds]),
                   np.concatenate([y_idx[v] for v, _ in folds]))
            del folds
        else:
            best = 0
            self.cv_score_ = None
            self.alpha_scores_ = None
            oos = (None, None)
        stats["t_cv"] = tt() - t2
        t3 = tt()
        op = _CenteredOp(D, np.arange(n), self._n_jobs)
        coef, icpt = _lsqr_solve(op, Y, (alphas[best],), group, CV_RIDGE_TOL)
        self.coef_ = np.ascontiguousarray(coef[0].T, dtype=np.float32)
        self.intercept_ = np.asarray(icpt[0], dtype=np.float64)
        self.alpha_ = float(alphas[best])
        self.solver_ = f"lsqr_cv{k}"
        self.n_epochs_ = None
        stats["t_solve"] = tt() - t3
        stats["lsqr_group"] = group
        stats["recomputed"] = int(design.recomputed)
        self._apply_prior(oos[0], oos[1], Y.mean(axis=0), stats)
