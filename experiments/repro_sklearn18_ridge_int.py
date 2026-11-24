"""Reproducer for the defect of Appendix E of the paper: scikit-learn 1.8's
RidgeClassifierCV fitted on integer-typed features predicts one class.

A synthetic two-class problem, 60 series x 400 count features (Poisson counts
at rate 0.3; the first 20 columns carry a class signal: Poisson(2) added for
class 1). The same matrix is fitted as int64 (sparse and dense) and as float64
(sparse). Under scikit-learn 1.8.0 the int64 fits give a training accuracy of
0.40 (one predicted class), the float64 fit 1.00.

BORF's counts are integers and gave the same symptom until cast to float
(experiments/borf_ridge.py casts); aeon 1.5's WEASEL+MUSE with its default
ridge head showed it too (cause not established there).

    python experiments/repro_sklearn18_ridge_int.py
"""
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import RidgeClassifierCV


def main():
    rng = np.random.RandomState(0)
    n, n_features = 60, 400
    y = rng.randint(0, 2, n)
    X = rng.poisson(0.3, (n, n_features))
    X[y == 1, :20] += rng.poisson(2.0, ((y == 1).sum(), 20))
    print("scikit-learn", sklearn.__version__)
    for name, M in [
        ("int64 sparse", sp.csr_matrix(X.astype(np.int64))),
        ("int64 dense", X.astype(np.int64)),
        ("float64 sparse", sp.csr_matrix(X.astype(np.float64))),
    ]:
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(M, y)
        pred = model.predict(M)
        print(f"{name:15s} train accuracy {(pred == y).mean():.2f}, "
              f"predicted classes {len(set(pred))}, alpha {model.alpha_:g}")


if __name__ == "__main__":
    main()
