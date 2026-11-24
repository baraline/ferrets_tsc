"""Values compiled into the kernels (not patchable) and values that fix the
model: word length, kinds and modes, the family draw's salts, the dummy
kernel arguments."""

import numpy as np


WORD_LEN = 8
KEYSPACE = 256
N_COEF = WORD_LEN // 2    # Fourier coefficients of a word: a real and an
                          # imaginary part, one bit each
MASK_WORDS = KEYSPACE // 64   # 64-bit words of the key occupancy mask


EPS_STD = 1e-8            # flat window: std <= this
RESCUE_CHAIN = 1e-6       # two-pass rescue: var <= this x the chain's energy
RESCUE_REL = 1e-8         # ... or var <= this x the window's energy

# Tie margins, added to the fitted cut tables by transform.cuts._tie_margins.
# They define what a stored table is: a comparison against a cut that two
# equal windows could decide either way is decided by the margin, not by the
# arithmetic.
TIE_FOURIER = 1e-6        # every Fourier cut c becomes c + this (absolute,
                          # in the part scale): a part within it above the
                          # sampled median falls on the 0 side, and a zero
                          # or residue cut turns positive, so the flat key
                          # bit (cut < 0) is 0 either way
TIE_MEAN = 1e-10          # level mean cut c_j becomes c_j + this x (|c_j| +
                          # (c_15 - c_1)) within its cut set
TIE_STD = 1e-5            # level std cut > 0 becomes c_j x (1 + this); cuts
                          # equal to 0 (the flat snap) stay 0


_U16_NONE = 65535         # "no kept column" in a uint16 head-column table
_KIND_PAA = 0             # PAA word
_KIND_FOURIER = 1        # Fourier word
_MODE_GLOBAL = 0
_MODE_SCORE = 1

# the multivariate (C >= 2) family layout
_SOURCES = ("paa", "fourier")   # the PAA and the Fourier word source
_FAMILY_KS = (2, 4, 8)      # channels per family (at most the largest power
                            # of two <= C)
_SUBSET_SALT = 1000003      # family channels: RandomState(seed * 7919 +
                            # this + side)
# cross-channel words as a bit code: bit 0 the coarse word, bit 1 the flags
_CROSS = {"none": 0, "coarse": 1, "flags": 2, "all": 3}
_CROSS_ALL = 3              # "all": the coarse word and the two flag words
_FAMILY_CUTS = 64           # cut slots per Fourier family: 8 per channel,
                            # 8 channels
LEVEL_BITS = 4              # bits of one level bin
LEVEL_CUTS = 15             # cuts per level bin search (16 bins): a cut set
                            # holds 15 mean cuts, then 15 std cuts. Fixed by
                            # kernels.level._bin15 and the 16 (j / 16) of
                            # kernels.cuts: 2**LEVEL_BITS - 1


def head_col_none(id_dtype):
    """The head-column table's value for a key with no kept column: _U16_NONE
    when the table's ids are uint16, else -1."""
    return _U16_NONE if np.dtype(id_dtype) == np.uint16 else -1


# kernel arguments that a mode does not use (never written)
_NO_IDS = np.zeros((1, 1), dtype=np.int32)
_NO_ACC = np.zeros((1, 1), dtype=np.float64)
_NO_VEC = np.zeros(1, dtype=np.float64)
_NO_IDF = np.zeros(1, dtype=np.float32)
_NO_W = np.zeros((1, 1), dtype=np.float32)
_NO_HC = np.zeros(0, dtype=np.uint16)
_NO_WT = np.zeros(0, dtype=np.float64)   # streaming head: no class weights
