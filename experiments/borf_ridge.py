"""BORF (Spinnato et al., IEEE Access 2024) under RidgeClassifierCV.

aeon 1.5.0 ships BORF as a transformation only
(aeon.transformations.collection.dictionary_based.BORF: SAX words at every
(window, word length, alphabet, dilation) of its grid, stride by the
window, deterministic, one sparse count column per receptive field). Its
paper pairs it with a ridge classifier; this pairs it with scikit-learn's
RidgeClassifierCV over alphas 1e-3..1e3, the head aeon gives MiniRocket.

Two choices of the wrapper:

* The counts are cast to float64 before the ridge. Under scikit-learn 1.8,
  RidgeClassifierCV fitted on integer-typed features predicts one class
  (experiments/repro_sklearn18_ridge_int.py, Appendix E of the paper).
* BORF's own parallelism is joblib over its configurations (``n_jobs``)
  with numba threads inside each (``n_jobs_numba``). The wrapper gives the
  thread budget to ``n_jobs`` and keeps numba at one thread per
  configuration (numba's thread start-up per configuration made the other
  split far slower), so the total stays within the budget.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV


class BORFRidge(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=1, random_state=None):
        self.n_jobs = n_jobs
        self.random_state = random_state   # BORF is deterministic; kept for
                                           # the drivers' uniform signature

    @staticmethod
    def _csr(F):
        return sp.csr_matrix(F).astype(np.float64)

    def fit(self, X, y):
        from aeon.transformations.collection.dictionary_based import BORF
        self.transformer_ = BORF(n_jobs=self.n_jobs, n_jobs_numba=1)
        F = self._csr(self.transformer_.fit_transform(X))
        self.n_features_ = F.shape[1]
        self.classifier_ = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10)).fit(F, y)
        return self

    def predict(self, X):
        return self.classifier_.predict(self._csr(self.transformer_.transform(X)))
