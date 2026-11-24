"""Column decoder and reference key scans for a fitted ``FERRETSClassifier``.

``FERRETSDecoder`` turns a head column back into the word it counts (view,
dilation, 8-bit pattern, window span; for a level word the mean and standard
deviation bins) and finds the word in series. ``reference_keys`` and
``reference_level_keys`` recompute every window's key with numpy in the
kernels' arithmetic. The decoder reads univariate models and the per-channel
layout, and refuses the family layout. See docs/model.md.
"""

from .decoder import FERRETSDecoder
from .reference import (ReferenceView, channel_series, nr_counts,
                        reference_keys, reference_level_keys, trig_tables)

__all__ = ["FERRETSDecoder", "ReferenceView", "reference_keys",
           "reference_level_keys", "nr_counts", "trig_tables",
           "channel_series"]
