"""Research switches for ablation studies, outside the public API.

``FERRETSAblation`` is a ``FERRETSClassifier`` with four more switches:
``head="logistic"`` (an SGD logistic head on the batch design), ``ridge_cv``
(k-fold cross-validated ridge alpha), ``prune_frac`` (keep the views with the
largest head weights and refit) and ``word_bits`` (4 or 2 bits per word). At
the defaults it is ``FERRETSClassifier``. The logistic head and ``ridge_cv``
refuse class weights: pass ``batch_class_weight=None``. See docs/model.md.
"""

from .estimator import FERRETSAblation
from .heads import CV_RIDGE_TOL

__all__ = ["FERRETSAblation", "CV_RIDGE_TOL"]
