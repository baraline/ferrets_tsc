"""The level word's per-window part: the bins of a window's mean and standard
deviation among a cut set's 15 sorted cuts (inline helpers of the keying
kernels)."""

import numpy as np
from numba import njit

from ..constants import EPS_STD, LEVEL_BITS, LEVEL_CUTS


@njit(inline="always")
def _bin15(cuts, row, off, x):
    """The number of cuts[row, off:off + 15] (sorted ascending) strictly
    below x, 0..15: a branchless four-step search."""
    j = 8 * np.int64(cuts[row, off + 7] < x)
    j += 4 * np.int64(cuts[row, off + j + 3] < x)
    j += 2 * np.int64(cuts[row, off + j + 1] < x)
    j += np.int64(cuts[row, off + j] < x)
    return j


@njit(inline="always")
def _level_part(cuts, row, mu, sd, hm, hs):
    """One channel's part of a window's level word: its mean bin's top hm
    bits above its std bin's top hs bits, the bins against cut set ``row``
    (15 mean cuts, then 15 std cuts).

    A flat window, one whose standard deviation is at most EPS_STD (the
    window the PAA and Fourier words key as flat), takes standard deviation
    0: its std bin is 0, the number of cuts below 0 (the std cuts are 0 or
    above EPS_STD), however the window's moments rounded. Otherwise the
    rounding residue of a window of equal values (the two-pass rescue in
    kernels.window can leave about 2e-15 times the window's level in the
    deviation) would be binned against the cuts, and constant windows of
    different levels would fall in different std bins."""
    mb = _bin15(cuts, row, 0, mu)
    sb = _bin15(cuts, row, LEVEL_CUTS, 0.0 if sd <= EPS_STD else sd)
    return ((mb >> (LEVEL_BITS - hm)) << hs) | (sb >> (LEVEL_BITS - hs))
