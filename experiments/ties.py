"""Do identical windows always receive identical keys, and how often does a
tie decide a bit? Untimed.

A FERRETS bit is a floating-point comparison computed by the keying kernel
(ferrets/kernels/; thresholds in ferrets/constants.py): a PAA bit compares a
block sum with an eighth of the window's block-sum total; a Fourier bit
compares a rotated, scaled coefficient with a training median plus a margin;
the level word bins the window mean and standard deviation among 15 training
quantiles each, with margins on the cuts and a flat window's std taken as 0.
Where a comparison is a tie in exact arithmetic (a block mean equal to the
window mean, frequent on quantised series: at w = 8 a block is one value),
rounding could decide the bit. This script measures it on the package
itself.

Per unit (a dataset at a fold): fit the shipped model
``ferrets.FERRETSClassifier(n_jobs=T, **FERRETS_PARAMS)`` (benchkit/arms.py;
batch head: the keys do not depend on the head) on the training split. Then,
for each split (train, test: keys from the model fitted on train) and each
block of the model -- a word block (one PAA or Fourier dictionary at one
dilation) or a level block (one (side, window, dilation), owned by the first
view keying it) -- the key the model assigns to every window of every series
is read with ``FERRETSDecoder.block_keys_many`` (ferrets/decode: the numpy
reference scan in the kernel's arithmetic, operation for operation), and:

(a) identical-window groups. The windows of one block (within a series and
    across the series of the split) whose w input values are bit-identical
    form a group. Grouping hashes each window with two independent 64-bit
    polynomial hashes of its mixed bit patterns (a 128-bit identity); a
    sorted pass finds the groups. A group of size >= 2 is inconsistent when
    its windows carry more than one key. Checks: windows of inconsistent
    groups (up to --verify-cap per block) and a sample of windows of
    consistent groups are compared byte for byte with their group's first
    window (group_check_fail: expected 0); the rolling hashes of a sample of
    windows are recomputed directly from their values (hash_check_fail:
    expected 0); adjacent windows with equal first hash and different second
    hash are counted (hash_collisions: expected 0).
(b) exact ties, on a random sample of up to --tie-sample windows per block
    (Python integers: every float64 is an integer times a power of two, so
    sums, squares and comparisons with the cuts are exact). PAA blocks: a
    window is tied when some block b has 8 x (sum of block b) == (sum of the
    window), among windows that are not flat in exact arithmetic
    (flat_rate_paa is the exact-flat fraction). Level blocks: a window is
    tied when its exact mean equals one of its mean cuts, or its exact
    standard deviation one of its std cuts. exact_mismatch_rate: the fraction
    of windows whose kernel key differs from the key evaluated in exact
    arithmetic ([x > cut], ties give 0). Rates are window-weighted across
    blocks. Fourier parts have no exact value (irrational rotations):
    Fourier blocks enter (a) only.
(c) the scan is the model: on the first --check-series series of each split
    the counts built from the scanned keys equal FERRETSDecoder.model_counts
    (the model's own kernel) for every view (count_check_fail: expected 0).

Units: the synthetic cases first (IntChain3_m512 and IntChain3_m500:
integer chains with three levels, a random walk with steps uniform in {-1,
0, +1} clipped to {0, 1, 2}, 20 + 20 series of length 512 or 500, two
alternating labels; besides the model's own dictionaries each carries fixed
PAA dictionaries at dilation 1 on the series side, one per window of the
grid, key_kind paa_fixed_w<w>_d1), then the 112 UCR datasets at fold 0, the
published split, read by benchkit.data.load_folds, smallest (n_train +
n_test) x m first, then the univariate MONSTER problems of --monster
(default Pedestrian and Traffic, integer-valued raw input) on the hub's
fold, each split a seeded random sample of at most --monster-max-series
series (the model is fitted on the train sample). MONSTER data are
downloaded from the Hugging Face hub when not cached; with --offline a
problem whose files are not cached is skipped with a message and no row.

Rows: results/ties.csv (--out), one per (benchmark, dataset, fold, split,
key_kind), key_kind in word (PAA and Fourier word blocks together),
word_paa, word_fourier, level, then the same counts per window length,
word_paa_w<w> and level_w<w>, and the fixed dictionaries paa_fixed_w<w>_d1
of the synthetic cases. A unit's rows are appended when the unit ends; a
failed fit is one row (split and key_kind empty, error = repr), a failed
split one row per split; every banked (benchmark, dataset, fold) is skipped
on a rerun (delete its rows to re-run it). Counts are sums over the row's
blocks; frac_* = count / n_windows; *_within: groups formed inside one
series only; n_windows_off_mode: windows of inconsistent groups outside
their group's most frequent key; fit_seconds and scan_seconds are wall times
of an untimed run.

usage:
  ties.py [--datasets all|none|D,E] [--folds 0] [--monster A,B|none]
        [--monster-max-series 2000] [--offline] [--threads 6]
        [--workers N] [--out PATH] [--dry-run]
The released rows ran at 6 threads.
"""

import os
import sys

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS")
DEFAULT_THREADS = 6


def _argv_value(flag, default):
    """``flag value`` or ``flag=value`` from sys.argv, before argparse."""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default


if __name__ == "__main__":
    # before numpy / numba / BLAS load
    for _var in THREAD_ENV:
        os.environ[_var] = str(_argv_value("--threads", DEFAULT_THREADS))
    if "--offline" in sys.argv:
        # before huggingface_hub loads: the MONSTER loader reads the cache only
        os.environ["HF_HUB_OFFLINE"] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchkit.arms import FERRETS_PARAMS  # noqa: E402
from experiments.benchkit import data as bdata  # noqa: E402

OUT_DEFAULT = REPO_ROOT / "results" / "ties.csv"
# m = 512: at dilation 1 the chain length is a power of two, so the kernel's
# chain mean, centred values and prefix sums are exact on these integers and
# its window means carry no rounding; m = 500: they do
SYNTHETIC = {"IntChain3_m512": dict(n_train=20, n_test=20, m=512, levels=3,
                                    seed=20261029),
             "IntChain3_m500": dict(n_train=20, n_test=20, m=500, levels=3,
                                    seed=20261029)}
MONSTER_DEFAULT = ("Pedestrian", "Traffic")
MODEL_KINDS = ("word", "word_paa", "word_fourier", "level")

COLUMNS = [
    "benchmark", "dataset", "fold", "split", "key_kind",
    "n_series", "m", "n_blocks", "n_windows", "n_distinct_windows",
    "n_groups", "n_windows_in_groups",
    "n_groups_inconsistent", "n_windows_in_inconsistent_groups",
    "frac_windows_in_inconsistent_groups",
    "n_windows_off_mode", "frac_windows_off_mode",
    "n_blocks_inconsistent",
    "n_groups_within", "n_groups_inconsistent_within",
    "n_windows_in_inconsistent_groups_within",
    "frac_windows_in_inconsistent_groups_within",
    "n_exact_sample", "tie_rate_paa", "flat_rate_paa",
    "tie_rate_level", "tie_rate_level_mean", "tie_rate_level_std",
    "exact_mismatch_rate",
    "hash_checks", "hash_check_fail", "hash_collisions",
    "group_checks", "group_check_fail",
    "count_check_series", "count_check_views", "count_check_fail",
    "frac_integer_values", "n_distinct_values",
    "fit_seconds", "scan_seconds", "note", "n_threads", "error",
]
# per-block integer counters, summed into a row
_SUM_KEYS = ("n_windows", "n_distinct_windows", "n_groups",
             "n_windows_in_groups", "n_groups_inconsistent",
             "n_windows_in_inconsistent_groups", "n_windows_off_mode",
             "n_groups_within", "n_groups_inconsistent_within",
             "n_windows_in_inconsistent_groups_within", "n_exact_sample",
             "hash_checks", "hash_check_fail", "hash_collisions",
             "group_checks", "group_check_fail")


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Exact arithmetic (Python integers)
# ---------------------------------------------------------------------------

def _dyadic(x):
    """(p, e) with x == p * 2**e exactly (x a finite float)."""
    fr = Fraction(float(x))
    return fr.numerator, -(fr.denominator.bit_length() - 1)


def exact_ints(U):
    """(ints, E) with U == ints * 2**E exactly: ints an object array of
    Python ints of U's shape (U finite float64)."""
    U = np.asarray(U, dtype=np.float64)
    mant, ex = np.frexp(U)
    mi = np.ldexp(mant, 53).astype(np.int64)        # |mi| < 2**53, exact
    ei = ex.astype(np.int64) - 53
    nz = mi != 0
    if not nz.any():
        return np.zeros(U.shape, dtype=np.int64).astype(object), 0
    E = int(ei[nz].min())
    sh = np.where(nz, ei - E, 0)
    return np.left_shift(mi.astype(object), sh.astype(object)), E


def _cmp(A, a_exp, b, b_exp):
    """(A * 2**a_exp > b * 2**b_exp, ... ==) elementwise and exact: A an
    object array of ints, b an int."""
    e = min(a_exp, b_exp)
    lhs = A * (1 << (a_exp - e)) if a_exp > e else A
    rhs = b << (b_exp - e) if b_exp > e else b
    return (np.asarray(lhs > rhs, dtype=bool),
            np.asarray(lhs == rhs, dtype=bool))


def _exact_moments(U):
    """(T, W2V, E, ints): the window sums T and w^2 x the variance in units
    of 2**(2E), W2V = w sum(x^2) - (sum x)^2, all exact."""
    w = U.shape[1]
    ints, E = exact_ints(U)
    T = ints.sum(axis=1)
    Q = (ints * ints).sum(axis=1)
    return T, w * Q - T * T, E, ints


def exact_paa(U, eps):
    """Definitions 3 and 5 in exact arithmetic for the PAA word of the
    windows U (k, w): (key, tied, flat). A window is flat when its variance
    is <= eps^2 (key 0); bit b is [8 x block sum b > window sum]; tied: some
    block b of a non-flat window has 8 x block sum == window sum."""
    k, w = U.shape
    paa = w // 8
    T, W2V, E, ints = _exact_moments(U)
    p, e = _dyadic(eps)
    above, _ = _cmp(W2V, 2 * E, w * w * p * p, 2 * e)
    flat = ~above
    B8 = ints.reshape(k, 8, paa).sum(axis=2) * 8
    gt = np.asarray(B8 > T[:, None], dtype=bool)
    tie = np.asarray(B8 == T[:, None], dtype=bool)
    bits = (gt.astype(np.int64) << np.arange(7, -1, -1)).sum(axis=1)
    key = np.where(flat, 0, bits)
    return key, tie.any(axis=1) & ~flat, flat


def exact_level(U, cuts, n_cuts):
    """The level word (k = 1: the mean bin's 4 bits above the std bin's) of
    the windows U (k, w) in exact arithmetic: (key, tied_mean, tied_std). A
    bin is the number of cuts strictly below the value (cuts: n_cuts mean
    cuts, then n_cuts std cuts); the std is compared through the variance
    (cut >= 0: std > cut iff var > cut^2)."""
    k, w = U.shape
    T, W2V, E, _ = _exact_moments(U)
    bm = np.zeros(k, dtype=np.int64)
    bs = np.zeros(k, dtype=np.int64)
    tm = np.zeros(k, dtype=bool)
    ts = np.zeros(k, dtype=bool)
    for c in np.asarray(cuts[:n_cuts], dtype=np.float64):
        if not np.isfinite(c):          # the kernel's cut < x
            bm += int(c == -np.inf)
            continue
        p, e = _dyadic(c)
        gt, eq = _cmp(T, E, w * p, e)
        bm += gt
        tm |= eq
    for c in np.asarray(cuts[n_cuts:2 * n_cuts], dtype=np.float64):
        if not np.isfinite(c):
            bs += int(c == -np.inf)
            continue
        if c < 0:
            bs += 1
            continue
        p, e = _dyadic(c)
        gt, eq = _cmp(W2V, 2 * E, w * w * p * p, 2 * e)
        bs += gt
        ts |= eq
    return bm * 16 + bs, tm, ts


# ---------------------------------------------------------------------------
# Window hashing (two independent 64-bit polynomial hashes of mixed bits)
# ---------------------------------------------------------------------------

_MASK = (1 << 64) - 1
_BASES = (0x9E3779B97F4A7C15, 0xD6E8FEB86659FD93)      # odd
_SALTS = (0x632BE59BD9B4E019, 0x85EBCA77C2B2AE63)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _mix(bits, salt):
    """splitmix64's finaliser of (bits + salt), elementwise, mod 2**64."""
    with np.errstate(over="ignore"):
        z = bits + np.uint64(salt)
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        return z ^ (z >> np.uint64(31))


def _powers(base, n):
    out = np.empty(n, dtype=np.uint64)
    x = 1
    for i in range(n):
        out[i] = x
        x = (x * base) & _MASK
    return out


class Hasher:
    """The powers of both bases and of their inverses mod 2**64, up to a
    series length m."""

    def __init__(self, m):
        self.binv = [_powers(pow(b, -1, 1 << 64), m + 1) for b in _BASES]
        self.bpow = [_powers(b, m + 1) for b in _BASES]

    def windows(self, A, h, w, d, nwin):
        """Hash h of every window (length w, dilation d) of the mixed rows A
        (n, m) uint64, window-start order (n, nwin): sum_k A[j + k d] x
        binv^k, from prefix sums of A[s] binv^s along each residue chain."""
        n, m = A.shape
        L = -(-m // d)
        if L * d == m:
            pad = A
        else:
            pad = np.zeros((n, L * d), dtype=np.uint64)
            pad[:, :m] = A
        with np.errstate(over="ignore"):
            Z = pad.reshape(n, L, d) * self.binv[h][:L, None]
            P = np.empty((n, L + 1, d), dtype=np.uint64)
            P[:, 0, :] = 0
            np.cumsum(Z, axis=1, out=P[:, 1:, :])
            del Z
            T = L - w + 1
            H = (P[:, w:w + T, :] - P[:, :T, :]) * self.bpow[h][:T, None]
        return H.reshape(n, T * d)[:, :nwin]

    def direct(self, Wbits, h):
        """Hash h of windows given by their bit patterns (k, w) uint64."""
        w = Wbits.shape[1]
        with np.errstate(over="ignore"):
            return (_mix(Wbits, _SALTS[h]) * self.binv[h][None, :w]).sum(
                axis=1, dtype=np.uint64)


def _gather(S, flat, nwin, w, d):
    """The values (len(flat), w) of the windows with flat index i nwin + j
    (series i, start j) of S (n, m), dilation d."""
    flat = np.asarray(flat, dtype=np.int64)
    i = flat // nwin
    j = flat % nwin
    return S[i[:, None], j[:, None] + d * np.arange(w)[None, :]]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _runs(new):
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, new.shape[0]))
    return starts, sizes


def _off_mode(gid_members, k_members):
    """Members outside their group's most frequent key (groups given by
    gid_members, keys by k_members)."""
    if gid_members.shape[0] == 0:
        return 0
    codes = gid_members.astype(np.int64) * 256 + k_members.astype(np.int64)
    uc, cnt = np.unique(codes, return_counts=True)
    g = uc // 256
    gs = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return int(gid_members.shape[0] - np.maximum.reduceat(cnt, gs).sum())


def group_stats(h1, h2, keys, nwin, Sbits, w, d, rng, verify_cap,
                verify_sample):
    """Identical-window groups of one block (flat window index i nwin + j):
    counts across the split and within each series, and the byte-for-byte
    verification of grouped windows."""
    N = h1.shape[0]
    out = dict(n_windows=N)
    order = np.argsort(h1)
    a = h1[order]
    b = h2[order]
    same1 = a[1:] == a[:-1]
    coll = int(np.count_nonzero(same1 & (b[1:] != b[:-1])))
    if coll:            # a 64-bit coincidence: sort on both hashes
        order = np.lexsort((h2, h1))
        a = h1[order]
        b = h2[order]
        same1 = a[1:] == a[:-1]
        coll = int(np.count_nonzero(same1 & (b[1:] != b[:-1])))
    out["hash_collisions"] = coll
    new = np.empty(N, dtype=bool)
    new[0] = True
    new[1:] = ~(same1 & (b[1:] == b[:-1]))
    del a, b, same1
    k = keys[order]
    starts, sizes = _runs(new)
    gid = np.cumsum(new) - 1
    del new
    multi = sizes >= 2
    kmin = np.minimum.reduceat(k, starts)
    kmax = np.maximum.reduceat(k, starts)
    incons = multi & (kmin != kmax)
    out["n_distinct_windows"] = int(starts.shape[0])
    out["n_groups"] = int(multi.sum())
    out["n_windows_in_groups"] = int(sizes[multi].sum())
    out["n_groups_inconsistent"] = int(incons.sum())
    out["n_windows_in_inconsistent_groups"] = int(sizes[incons].sum())
    pos_inc = np.flatnonzero(incons[gid])
    out["n_windows_off_mode"] = _off_mode(gid[pos_inc], k[pos_inc])
    # within one series: the groups of (window, series) inside the groups
    pos = np.flatnonzero(multi[gid])
    zero = dict(n_groups_within=0, n_groups_inconsistent_within=0,
                n_windows_in_inconsistent_groups_within=0)
    out.update(zero)
    if pos.shape[0]:
        ser = (order[pos] // nwin).astype(np.uint64)
        comp = (gid[pos].astype(np.uint64) << np.uint64(32)) | ser
        o2 = np.argsort(comp)
        c2 = comp[o2]
        k2 = k[pos][o2]
        new2 = np.empty(c2.shape[0], dtype=bool)
        new2[0] = True
        new2[1:] = c2[1:] != c2[:-1]
        st2, sz2 = _runs(new2)
        m2 = sz2 >= 2
        inc2 = m2 & (np.minimum.reduceat(k2, st2)
                     != np.maximum.reduceat(k2, st2))
        out["n_groups_within"] = int(m2.sum())
        out["n_groups_inconsistent_within"] = int(inc2.sum())
        out["n_windows_in_inconsistent_groups_within"] = int(
            sz2[inc2].sum())
    # verification: each checked window against its group's first window
    checks = []
    if pos_inc.shape[0] > verify_cap:
        checks.append(rng.choice(pos_inc, size=verify_cap, replace=False))
    else:
        checks.append(pos_inc)
    pos_ok = np.flatnonzero((multi & ~incons)[gid])
    if pos_ok.shape[0] > verify_sample:
        pos_ok = rng.choice(pos_ok, size=verify_sample, replace=False)
    checks.append(pos_ok)
    n_chk = n_fail = 0
    for p in checks:
        for s in range(0, p.shape[0], 20000):
            q = p[s:s + 20000]
            first = order[starts[gid[q]]]
            A = _gather(Sbits, order[q], nwin, w, d)
            B = _gather(Sbits, first, nwin, w, d)
            n_chk += q.shape[0]
            n_fail += int(np.count_nonzero(~np.all(A == B, axis=1)))
    out["group_checks"] = n_chk
    out["group_check_fail"] = n_fail
    return out


# ---------------------------------------------------------------------------
# One block
# ---------------------------------------------------------------------------

def analyse_block(blk, X2, S, Sbits, A, hasher, args, seed, n_check, eps,
                  n_cuts):
    """Keys, groups, exact sample and check keys of one block over the
    split's series X2 (n, m); S the block's side (n, m), Sbits its bit
    patterns, A its two mixed copies."""
    n, m = S.shape
    w, d = blk["w"], blk["d"]
    nwin = m - (w - 1) * d
    zeros = {k: 0 for k in _SUM_KEYS}
    res = dict(zeros, w_tie=0.0, w_flat=0.0, w_tm=0.0, w_ts=0.0,
               w_mismatch=0.0, w_weight=0)
    nc = min(n_check, n)
    if nwin <= 0 or n == 0:
        res["check_keys"] = np.zeros((nc, 0), dtype=np.int64)
        return res
    N = n * nwin
    h1 = np.empty(N, dtype=np.uint64)
    h2 = np.empty(N, dtype=np.uint64)
    keys = np.empty(N, dtype=np.uint8)
    rows = max(1, int(args.chunk_values // m))
    for c0 in range(0, n, rows):
        c1 = min(n, c0 + rows)
        kk = blk["keys"](X2[c0:c1])
        if kk.shape != (c1 - c0, nwin):
            raise ValueError(f"block {blk['name']}: keys {kk.shape}, "
                             f"expected {(c1 - c0, nwin)}")
        if kk.size and (kk.min() < 0 or kk.max() > 255):
            raise ValueError(f"block {blk['name']}: a key outside 0..255")
        sl = slice(c0 * nwin, c1 * nwin)
        keys[sl] = kk.ravel()
        h1[sl] = hasher.windows(A[0][c0:c1], 0, w, d, nwin).ravel()
        h2[sl] = hasher.windows(A[1][c0:c1], 1, w, d, nwin).ravel()
        del kk
    rng = np.random.default_rng(seed)
    res.update(group_stats(h1, h2, keys, nwin, Sbits, w, d, rng,
                           args.verify_cap, args.verify_sample))
    # the rolling hashes against a direct recomputation from the values
    hs = rng.choice(N, size=min(N, args.hash_sample), replace=False)
    Wb = _gather(Sbits, hs, nwin, w, d)
    res["hash_checks"] = int(hs.shape[0])
    res["hash_check_fail"] = int(np.count_nonzero(
        (hasher.direct(Wb, 0) != h1[hs]) | (hasher.direct(Wb, 1) != h2[hs])))
    del h1, h2
    # the exact sample
    if blk["exact"] is not None:
        idx = np.sort(rng.choice(N, size=min(N, args.tie_sample),
                                 replace=False))
        U = _gather(S, idx, nwin, w, d)
        kern = keys[idx].astype(np.int64)
        if blk["exact"] == "paa":
            ex_key, tied, flat = exact_paa(U, eps)
            res["w_flat"] = float(flat.mean()) * N
        else:
            ex_key, tm, ts = exact_level(U, blk["cuts"], n_cuts)
            tied = tm | ts
            res["w_tm"] = float(tm.mean()) * N
            res["w_ts"] = float(ts.mean()) * N
        res["n_exact_sample"] = int(idx.shape[0])
        res["w_tie"] = float(tied.mean()) * N
        res["w_mismatch"] = float((ex_key != kern).mean()) * N
        res["w_weight"] = N
    res["check_keys"] = keys[:nc * nwin].reshape(nc, nwin).astype(np.int64)
    return res


# ---------------------------------------------------------------------------
# A unit
# ---------------------------------------------------------------------------

def synthetic_split(name, fold):
    """A synthetic case (SYNTHETIC): random walks with steps uniform in {-1,
    0, +1}, clipped to {0, ..., levels - 1}, starting uniformly, as (n, 1, m)
    float64; labels alternate 0, 1; the generator is seeded by (seed,
    fold)."""
    p = SYNTHETIC[name]
    rng = np.random.default_rng([p["seed"], int(fold)])
    m, top = p["m"], p["levels"] - 1

    def make(n):
        x = np.empty((n, m), dtype=np.float64)
        x[:, 0] = rng.integers(0, top + 1, size=n)
        steps = rng.integers(-1, 2, size=(n, m - 1))
        for t in range(1, m):
            x[:, t] = np.clip(x[:, t - 1] + steps[:, t - 1], 0, top)
        return x[:, None, :], (np.arange(n) % 2).astype(np.int64)
    X_tr, y_tr = make(p["n_train"])
    X_ts, y_ts = make(p["n_test"])
    note = (f"random walk, steps uniform in {{-1,0,1}}, clipped to "
            f"{{0..{top}}}, seed {p['seed']}")
    return X_tr, y_tr, X_ts, y_ts, note


def load_unit(unit, args):
    bench, name, fold = unit["benchmark"], unit["dataset"], unit["fold"]
    if bench == "synthetic":
        return synthetic_split(name, fold)
    f = next(iter(bdata.load_folds(bench, name, [fold])))
    X_tr, y_tr, X_ts, y_ts = f.X_train, f.y_train, f.X_test, f.y_test
    note = ""
    if bench == "monster":
        cap = int(args.monster_max_series)
        rng = np.random.default_rng([int(args.seed), int(fold)])
        parts = []
        out = []
        for X, y, side in ((X_tr, y_tr, "train"), (X_ts, y_ts, "test")):
            n = X.shape[0]
            if n > cap:
                idx = np.sort(rng.choice(n, size=cap, replace=False))
                X, y = np.ascontiguousarray(X[idx]), y[idx]
            parts.append(f"{side} sample {X.shape[0]} of {n}")
            out += [X, y]
        X_tr, y_tr, X_ts, y_ts = out
        note = "; ".join(parts) + f"; sample seed [{args.seed}, {fold}]"
    return X_tr, y_tr, X_ts, y_ts, note


def model_blocks(dec):
    """Every word block and level block of a fitted model, plan order."""
    from ferrets.constants import LEVEL_CUTS
    blocks = []
    for v, view in enumerate(dec.views):
        w = view.window
        side = int(dec.channel[v])
        kind = "word_paa" if view.kind == "paa" else "word_fourier"
        for di, d in enumerate(dec.dils[v]):
            def keys(Xc, v=v, d=d):
                return dec.block_keys_many(Xc, v, d)
            blocks.append(dict(
                name=f"v{v}:{view.kind}{w}:d{d}", v=v, di=di, w=w, d=int(d),
                side=side, kind=kind, level=False, keys=keys,
                exact="paa" if view.kind == "paa" else None, cuts=None))
            if dec.slot_level[v][di]:
                def lkeys(Xc, v=v, d=d):
                    return dec.block_keys_many(Xc, v, d, level=True)
                cuts = np.asarray(view.level_cuts, dtype=np.float64)
                if cuts.shape != (2 * LEVEL_CUTS,):
                    raise ValueError(f"view {v}: level cut set {cuts.shape}")
                blocks.append(dict(
                    name=f"v{v}:level{w}:d{d}", v=v, di=di, w=w, d=int(d),
                    side=side, kind="level", level=True, keys=lkeys,
                    exact="level", cuts=cuts))
    return blocks


def fixed_blocks(m):
    """The synthetic case's fixed PAA dictionaries: dilation 1, the series
    side, every window of the grid that fits."""
    from ferrets.decode import reference_keys
    out = []
    for w in FERRETS_PARAMS["windows"]:
        if w > m:
            continue

        def keys(Xc, w=w):
            return reference_keys(Xc, "paa", w, 1)
        out.append(dict(name=f"fixed:paa{w}:d1", v=-1, di=-1, w=int(w), d=1,
                        side=0, kind=f"paa_fixed_w{w}_d1", level=False,
                        keys=keys, exact="paa", cuts=None))
    return out


def _kinds_of(blk):
    """The key kinds (rows) a block counts in: its aggregate and, for a PAA
    or level block of the model, its window's row."""
    kind = blk["kind"]
    if kind == "word_paa":
        return ("word", "word_paa", f"word_paa_w{blk['w']}")
    if kind == "word_fourier":
        return ("word", "word_fourier")
    if kind == "level":
        return ("level", f"level_w{blk['w']}")
    return (kind,)                       # a fixed dictionary


def _rows_of(blocks, results, base):
    """One row per key_kind from the block results: word, word_paa,
    word_fourier, level, then word_paa_w<w> and level_w<w> (ascending w),
    then the fixed dictionaries in block order."""
    groups = {k: [] for k in MODEL_KINDS}
    wins = sorted({b["w"] for b in blocks if b["kind"] in ("word_paa",
                                                            "level")})
    for prefix in ("word_paa", "level"):
        for w in wins:
            groups[f"{prefix}_w{w}"] = []
    for blk, r in zip(blocks, results):
        for kind in _kinds_of(blk):
            groups.setdefault(kind, []).append((blk, r))
    rows = []
    for kind, members in groups.items():
        if not members and kind not in MODEL_KINDS:
            continue
        row = dict(base, key_kind=kind, n_blocks=len(members))
        for k in _SUM_KEYS:
            row[k] = int(sum(r[k] for _, r in members))
        row["n_blocks_inconsistent"] = int(sum(
            r["n_groups_inconsistent"] > 0 for _, r in members))
        nw = row["n_windows"]
        for count, frac in (("n_windows_in_inconsistent_groups",
                             "frac_windows_in_inconsistent_groups"),
                            ("n_windows_off_mode", "frac_windows_off_mode"),
                            ("n_windows_in_inconsistent_groups_within",
                             "frac_windows_in_inconsistent_groups_within")):
            row[frac] = row[count] / nw if nw else None
        wt = sum(r["w_weight"] for _, r in members)
        has_exact = wt > 0 and kind != "word"

        def rate(key):
            return sum(r[key] for _, r in members) / wt
        is_level = kind.startswith("level")
        row["tie_rate_paa"] = rate("w_tie") if has_exact and not is_level \
            else None
        row["flat_rate_paa"] = rate("w_flat") if has_exact and not is_level \
            else None
        row["tie_rate_level"] = rate("w_tie") if has_exact and is_level \
            else None
        row["tie_rate_level_mean"] = rate("w_tm") if has_exact and is_level \
            else None
        row["tie_rate_level_std"] = rate("w_ts") if has_exact and is_level \
            else None
        row["exact_mismatch_rate"] = rate("w_mismatch") if has_exact \
            else None
        if kind == "word":
            row["n_exact_sample"] = 0
        rows.append(row)
    return rows


def count_check(dec, X2, blocks, results, n_check, nr):
    """Counts from the scanned keys of the first n_check series against the
    model's own MODE_GLOBAL counts, view by view: (series, views, fails)."""
    from ferrets.decode import nr_counts
    nc = min(n_check, X2.shape[0])
    if nc == 0:
        return 0, 0, 0
    by = {(b["v"], b["di"], b["level"]): r["check_keys"]
          for b, r in zip(blocks, results) if b["v"] >= 0}
    X3 = np.ascontiguousarray(X2[:nc, None, :])
    fails = 0
    for v in range(len(dec.views)):
        parts = []
        for di in range(len(dec.dils[v])):
            parts.append(nr_counts(by[(v, di, False)], nr=nr))
            if dec.slot_level[v][di]:
                parts.append(nr_counts(by[(v, di, True)], nr=nr))
        mine = np.concatenate(parts, axis=1)
        fails += int(not np.array_equal(mine, dec.model_counts(X3, v)))
    return nc, len(dec.views), fails


def scan_split(dec, model, X2, blocks, args, seed_base, n_check):
    """Every block of one split, a thread pool over the blocks; returns the
    block results and the count check."""
    from ferrets.constants import EPS_STD, LEVEL_CUTS
    from ferrets.decode import channel_series
    n, m = X2.shape
    sides = {}
    for s in sorted({b["side"] for b in blocks}):
        S = np.ascontiguousarray(channel_series(X2, bool(s)))
        bits = S.view(np.uint64)
        sides[s] = (S, bits, (_mix(bits, _SALTS[0]), _mix(bits, _SALTS[1])))
    hasher = Hasher(m)
    # the trig tables of every Fourier view (cached on the view per length),
    # built before the threads start
    warm = set()
    for b in blocks:
        if b["v"] >= 0 and b["kind"] == "word_fourier" and b["v"] not in warm \
                and n > 0 and m - (b["w"] - 1) * b["d"] > 0:
            dec.block_keys_one(X2[0], b["v"], b["d"])
            warm.add(b["v"])

    def run(item):
        bi, blk = item
        S, bits, A = sides[blk["side"]]
        return analyse_block(blk, X2, S, bits, A, hasher, args,
                             seed_base + [bi], n_check, EPS_STD, LEVEL_CUTS)
    workers = max(1, int(args.workers or args.threads))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run, enumerate(blocks)))
    nr = bool(model.transformer_._plan["numerosity_reduction"])
    chk = count_check(dec, X2, blocks, results, n_check, nr)
    return results, chk


def run_unit(unit, args):
    """Fit, then both splits; returns the unit's rows."""
    import ferrets
    from ferrets.decode import FERRETSDecoder
    bench, name, fold = unit["benchmark"], unit["dataset"], unit["fold"]
    T = int(args.threads)
    base0 = dict(benchmark=bench, dataset=name, fold=fold, n_threads=T)
    try:
        X_tr, y_tr, X_ts, y_ts, note = load_unit(unit, args)
        if X_tr.ndim == 3 and X_tr.shape[1] != 1:
            raise ValueError(f"{name}: {X_tr.shape[1]} channels; the decoder "
                             "reads univariate models only")
        t0 = time.perf_counter()
        model = ferrets.FERRETSClassifier(n_jobs=T, **FERRETS_PARAMS)
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        dec = FERRETSDecoder(model)
    except Exception as exc:                       # noqa: BLE001
        traceback.print_exc()
        say(f"  {bench} {name} fold {fold}: unit FAILED: {exc!r}")
        return [dict(base0, split="", key_kind="", error=_one_line(exc))]
    m = int(X_tr.shape[-1])
    blocks = model_blocks(dec)
    if bench == "synthetic":
        blocks += fixed_blocks(m)
    say(f"  {bench} {name} fold {fold}: fit {fit_s:.1f} s, "
        f"{len(dec.views)} views, {len(blocks)} blocks")
    rows = []
    salt = zlib.crc32(f"{bench}:{name}".encode())
    for si, (split, X) in enumerate((("train", X_tr), ("test", X_ts))):
        t1 = time.perf_counter()
        X2 = np.ascontiguousarray(np.asarray(X, dtype=np.float64)
                                  .reshape(X.shape[0], -1))
        base = dict(base0, split=split, n_series=int(X2.shape[0]), m=m,
                    note=note, fit_seconds=round(fit_s, 2))
        try:
            fin = X2[np.isfinite(X2)]
            base["frac_integer_values"] = (float(np.mean(fin == np.round(fin)))
                                           if fin.size else None)
            base["n_distinct_values"] = int(np.unique(X2).shape[0])
            del fin
            results, (cs, cv, cf) = scan_split(
                dec, model, X2, blocks, args,
                [int(args.seed), salt, int(fold), si], args.check_series)
            base.update(count_check_series=cs, count_check_views=cv,
                        count_check_fail=cf,
                        scan_seconds=round(time.perf_counter() - t1, 2))
            split_rows = _rows_of(blocks, results, base)
            rows += split_rows
            by = {r["key_kind"]: r for r in split_rows}
            say(f"  {split}: {X2.shape[0]} x {m}, "
                f"{by['word']['n_windows'] + by['level']['n_windows']} "
                f"window keys, word incons "
                f"{_fmt(by['word']['frac_windows_in_inconsistent_groups'])}, "
                f"level incons "
                f"{_fmt(by['level']['frac_windows_in_inconsistent_groups'])}"
                f", paa tie {_fmt(by['word_paa']['tie_rate_paa'])}, level "
                f"tie {_fmt(by['level']['tie_rate_level'])}, checks: counts "
                f"{cf} fail / {cv} views, hash "
                f"{sum(r['hash_check_fail'] for r in results)} fail, "
                f"collisions {sum(r['hash_collisions'] for r in results)}, "
                f"group {sum(r['group_check_fail'] for r in results)} fail "
                f"({time.perf_counter() - t1:.1f} s)")
        except Exception as exc:                   # noqa: BLE001
            traceback.print_exc()
            say(f"  {split}: FAILED: {exc!r}")
            rows.append(dict(base, key_kind="", error=_one_line(exc),
                             scan_seconds=round(time.perf_counter() - t1, 2)))
    return rows


def _fmt(x):
    return "-" if x is None else f"{x:.4g}"


def _one_line(exc):
    return " ".join(repr(exc).split())[:1000]


# ---------------------------------------------------------------------------
# Plan, bank, main
# ---------------------------------------------------------------------------

def _parse_folds(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ucr_names():
    """The 112 UCR datasets, smallest (n_train + n_test) x m first."""
    info = bdata._ucr_info()
    names = bdata.datasets("ucr")
    return sorted(names, key=lambda d: ((info[d]["training samples"]
                                         + info[d]["testing samples"])
                                        * info[d]["length"], d))


def monster_cached(name, fold):
    """(cached?, missing files) of one MONSTER problem's fold in the hub
    cache, without contacting the hub: the files of the snapshot refs/main
    names, as the offline loader resolves them. (Not try_to_load_from_cache:
    on this case-insensitive disk the loader's .no_exist marker for
    <name>_Y.npy also answers for the <name>_y.npy that exists.)"""
    from experiments.benchkit.reap import MONSTER_REPO, hub_cache
    repo = hub_cache() / MONSTER_REPO.format(name=name)
    ref = repo / "refs" / "main"
    snap = (repo / "snapshots" / ref.read_text().strip()
            if ref.is_file() else None)
    listed = set(os.listdir(snap)) if snap is not None and snap.is_dir() \
        else set()

    def have(fn):
        return fn in listed and (snap / fn).exists()   # the blob is there
    missing = [fn for fn in (f"{name}_X.npy",
                             f"test_indices_fold_{fold}.txt")
               if not have(fn)]
    if not (have(f"{name}_Y.npy") or have(f"{name}_y.npy")):
        missing.append(f"{name}_Y.npy|{name}_y.npy")
    return not missing, missing


def plan_units(args):
    folds = _parse_folds(args.folds)
    units = []
    ds = str(args.datasets).strip()
    if ds.lower() != "none":
        ucr = ucr_names()
        if ds.lower() == "all":
            names = list(SYNTHETIC) + ucr
        else:
            names = [d.strip() for d in ds.split(",") if d.strip()]
            bad = [d for d in names if d not in SYNTHETIC and d not in ucr]
            if bad:
                raise SystemExit(f"unknown datasets {bad}: the 112 UCR "
                                 f"names or {list(SYNTHETIC)}")
        for d in names:
            bench = "synthetic" if d in SYNTHETIC else "ucr"
            if bench == "synthetic":
                p = SYNTHETIC[d]
                size = (p["n_train"], p["n_test"], p["m"])
            else:
                n_tr, n_ts, _, m = bdata.sizes("ucr", d)
                size = (n_tr, n_ts, m)
            for f in folds:
                units.append(dict(benchmark=bench, dataset=d, fold=int(f),
                                  size=size, skip=None))
    if str(args.monster).strip().lower() != "none":
        names = [d.strip() for d in str(args.monster).split(",") if d.strip()]
        for d in names:
            if d not in bdata.MONSTER_SHAPES:
                raise SystemExit(f"unknown MONSTER problem {d!r}")
            n, C, m = bdata.MONSTER_SHAPES[d]
            cap = int(args.monster_max_series)
            size = (min(n - n // 5, cap), min(n // 5, cap), m)
            for f in folds:
                skip = None
                if C != 1:
                    skip = (f"multivariate (C = {C}): the decoder reads "
                            "univariate models only")
                elif args.offline:
                    ok, missing = monster_cached(d, f)
                    if not ok:
                        skip = ("not in the Hugging Face hub cache (missing "
                                f"{', '.join(missing)}): a download would "
                                "be needed (drop --offline)")
                units.append(dict(benchmark="monster", dataset=d,
                                  fold=int(f), size=size, skip=skip))
    return units


def banked_units(path):
    """(benchmark, dataset, fold) of every row already in the CSV; refuses a
    file written with other columns."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{path} has other columns than this driver "
                             "writes; use another --out")
        return {(r["benchmark"], r["dataset"], str(r["fold"]))
                for r in reader}


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k))
                         for k in COLUMNS})


def build_parser():
    p = argparse.ArgumentParser(
        description="Ties and key consistency of identical windows, on the "
                    "shipped FERRETS model (untimed).")
    p.add_argument("--datasets", default="all",
                   help="'all' (the synthetic cases + the 112 UCR "
                        "datasets), 'none', or a comma list of UCR names and "
                        f"synthetic cases ({', '.join(SYNTHETIC)})")
    p.add_argument("--folds", default="0",
                   help="comma list / ranges (default 0: the published "
                        "split; MONSTER: the hub's fold)")
    p.add_argument("--monster", default=",".join(MONSTER_DEFAULT),
                   help="comma list of univariate MONSTER problems, or none "
                        f"(default {','.join(MONSTER_DEFAULT)})")
    p.add_argument("--monster-max-series", type=int, default=2000,
                   help="series per MONSTER split (seeded random sample)")
    p.add_argument("--offline", action="store_true",
                   help="read MONSTER from the Hugging Face hub cache only "
                        "(uncached problems are skipped)")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="the thread budget: estimator n_jobs, numba / BLAS "
                        "threads, and the block pool's size")
    p.add_argument("--workers", type=int, default=None,
                   help="block pool size (default --threads; lower it to "
                        "cap memory)")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tie-sample", type=int, default=2000,
                   help="windows per block for the exact tests")
    p.add_argument("--check-series", type=int, default=4,
                   help="series per split for the count check")
    p.add_argument("--verify-cap", type=int, default=20000,
                   help="windows of inconsistent groups verified per block")
    p.add_argument("--verify-sample", type=int, default=2000,
                   help="windows of consistent groups verified per block")
    p.add_argument("--hash-sample", type=int, default=1000,
                   help="windows per block whose hashes are recomputed")
    p.add_argument("--chunk-values", type=float, default=1e6,
                   help="values per chunk of series in the reference scan")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    units = plan_units(args)
    banked = banked_units(out)
    todo = [u for u in units if u["skip"] is None
            and (u["benchmark"], u["dataset"], str(u["fold"])) not in banked]
    say(f"ties: {len(units)} units, {len(todo)} to run, threads "
        f"{args.threads}, workers {args.workers or args.threads}, out {out}")
    for i, u in enumerate(units, 1):
        n_tr, n_ts, m = u["size"]
        key = (u["benchmark"], u["dataset"], str(u["fold"]))
        status = ("SKIP: " + u["skip"] if u["skip"] else
                  "banked" if key in banked else "run")
        if args.dry_run or u["skip"]:
            print(f"  {i:3d} {u['benchmark']:9s} {u['dataset']:30s} fold "
                  f"{u['fold']}  n_train {n_tr:6d}  n_test {n_ts:6d}  m "
                  f"{m:5d}  values {(n_tr + n_ts) * m:10d}  {status}",
                  flush=True)
    if args.dry_run:
        unbanked = [u for u in units if (u["benchmark"], u["dataset"],
                                         str(u["fold"])) not in banked]
        print(f"cells to run: {len(unbanked)}", flush=True)
        return 0
    t_all = time.perf_counter()
    failed = 0
    for i, u in enumerate(todo, 1):
        t0 = time.perf_counter()
        say(f"{i}/{len(todo)} {u['benchmark']} {u['dataset']} fold "
            f"{u['fold']}: start")
        try:
            rows = run_unit(u, args)
        except Exception as exc:                   # noqa: BLE001
            traceback.print_exc()
            rows = [dict(benchmark=u["benchmark"], dataset=u["dataset"],
                         fold=u["fold"], split="", key_kind="",
                         n_threads=int(args.threads), error=_one_line(exc))]
        errs = sum(bool(r.get("error")) for r in rows)
        failed += errs > 0
        append_rows(out, rows)
        say(f"{i}/{len(todo)} {u['benchmark']} {u['dataset']} fold "
            f"{u['fold']}: {len(rows)} rows banked, {errs} error rows, "
            f"{time.perf_counter() - t0:.1f} s (total "
            f"{time.perf_counter() - t_all:.0f} s)")
    skipped = [u for u in units if u["skip"]]
    say(f"ties: {'COMPLETE' if not failed and not skipped else 'DONE'}"
        f" -- {len(todo)} units run, {failed} with errors, {len(skipped)} "
        f"skipped (no row), {time.perf_counter() - t_all:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
