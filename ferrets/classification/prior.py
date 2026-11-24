"""The prior shift of the batch heads."""

import math

import numpy as np

from .. import config
from ..classification.common import _decide, _eq_score


def _prior_lambda(dec, y, ybar, n_classes):
    """The prior shift's lambda from out-of-sample decisions dec (n, K') of
    rows labelled y: the smallest lambda of PRIOR_LAMBDAS whose equal-weight
    score of _decide(dec - lambda * ybar) is within one binomial standard
    error sqrt(s (1 - s) / n) of the best score s. Returns (lambda, scores)."""
    sc = [_eq_score(y, _decide(dec - lam * ybar[None, :]), n_classes)
          for lam in config.PRIOR_LAMBDAS]
    best = max(sc)
    se = math.sqrt(max(best * (1.0 - best), 1e-12) / len(y))
    for lam, v in zip(config.PRIOR_LAMBDAS, sc):
        if v >= best - se:
            return float(lam), tuple(float(v) for v in sc)
