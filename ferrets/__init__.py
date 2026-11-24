"""FERRETS: Fast Exact Random Radix Encodings of Time Series.

A symbolic transform that turns each series into a sparse bag of exact 8-bit
words, and a classifier on it. Two aeon estimators:

``FERRETS``
    The label-free transform (a collection transformer): random dictionaries
    key every dilated window of the series and of its first difference into
    256 words; ``transform`` returns the weighted bag rows as a scipy CSR
    matrix.
``FERRETSClassifier``
    The transform plus a linear head: an exact ridge (batch) or an averaged
    least-squares SGD (streaming, for large data).

>>> from ferrets import FERRETS, FERRETSClassifier
>>> clf = FERRETSClassifier(n_jobs=4).fit(X_train, y_train)  # doctest: +SKIP
>>> y_pred = clf.predict(X_test)  # doctest: +SKIP
>>> rows = FERRETS(n_jobs=4).fit_transform(X_train)  # doctest: +SKIP

The model is described in docs/model.md. Settings are in ``ferrets.config``.
"""

from .classifier import FERRETSClassifier
from .transformer import FERRETS

__all__ = ["FERRETS", "FERRETSClassifier"]
