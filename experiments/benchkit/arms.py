"""The arm registry: every classifier the benchmark runs, its parameters named.

Every construction names the parameters the measured configuration depends
on, so a later change of a library default does not change an arm. Builders
import lazily, so the parent process never loads torch or numba.
"""

import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
CONTRACT_MINUTES = 50
# the cost benchmark: time and memory only, declared on every arm so a cost
# run can compare all of them
COST = frozenset({"cost"})
ALL = frozenset({"ucr", "multiverse", "monster", "synthetic"}) | COST
_MV = frozenset({"multiverse"})


@dataclass(frozen=True)
class Arm:
    name: str
    benchmarks: frozenset
    build: Callable[[int], object]
    wsl: bool = False   # the child runs in a separate (WSL2) interpreter
    slow: bool = False  # expensive: gated, and scheduled after the fast arms
    univariate: bool = False  # the aeon estimator accepts one channel only
    # the child's JIT / thread-pool warm-up for this arm, called as
    # warm(n_threads, X, y) before any timer; None = one fit + predict on
    # the warm-up collection (child._warm_up)
    warm: Callable = None


def _repo_on_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


# ferrets.FERRETSClassifier's constructor, every parameter named: the
# package's defaults (the transform's parameters, then the head's).
FERRETS_PARAMS = dict(n_dictionaries=100,
                      windows=(8, 16, 24, 32, 48, 64), min_doc_freq=1,
                      alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
                      design_ram_mb=8192.0, random_state=0,
                      prior_shift=True, stream_class_weight="sqrt",
                      use_idf=False, numerosity_reduction=False,
                      cross_channel_words="all", channel_layout="family",
                      use_first_difference=True,
                      word_sources=("paa", "fourier"),
                      dilation="log_uniform", sublinear_tf=True,
                      level_word=True, batch_class_weight="sqrt")


def _ferrets(n_threads, use_streaming=False, **over):
    """FERRETS: ferrets.FERRETSClassifier with FERRETS_PARAMS. Multichannel
    input (C >= 2) is keyed into channel-wise and cross-channel words
    (families of 2, 4 or 8 channels on one dictionary); C = 1 is the
    univariate model.

    ``use_streaming`` False: batch fit, ridge head (GCV on the dual up to
    3000 training series, LSQR with a 25% hold-out above; sqrt class weights
    beside the prior shift), the UCR and Multiverse configuration. True:
    streaming fit, averaged least-squares SGD head over a compact chunk
    cache, the MONSTER configuration. ``n_jobs`` is the one thread budget:
    ferrets sets numba's count and limits BLAS to it for the length of fit
    and predict."""
    _repo_on_path()
    import ferrets
    params = dict(FERRETS_PARAMS)
    params.update(over)
    return ferrets.FERRETSClassifier(use_streaming=use_streaming,
                                     n_jobs=n_threads, **params)


def _ferrets_perchannel(n_threads, use_streaming=False, **over):
    """The per-channel layout (every channel and its first difference a side
    of its own, 100 draws each): FERRETSClassifier(channel_layout=
    "per_channel"), every other parameter FERRETS's."""
    return _ferrets(n_threads, use_streaming,
                    **dict(over, channel_layout="per_channel"))


def _gate_collection(seed=0):
    """36 univariate series of 256 points in 6 classes: the smallest
    collection that reaches the size-gated kernels the 24 x C x 96, 3-class
    warm-up misses (the LSQR route's transposed product with more than 4
    classes on rows of >= 1024 entries, and a df pass over >= 2 x 65536
    entries)."""
    import numpy as np
    X = np.cumsum(np.random.RandomState(seed).randn(36, 1, 256), axis=-1)
    y = (np.arange(36) % 6).astype(np.int64)
    return X, y


def _imbalanced(y):
    """The warm-up labels with unequal class sizes and the same number of
    classes: the batch head's class weights are None on balanced classes, and
    the weighted kernels they reach would compile inside the first timed fit
    of an imbalanced cell (24 rows, 3 classes: 6, 6, 12; 36 rows, 6 classes:
    6, 5, 5, 5, 5, 10)."""
    import numpy as np
    y = np.asarray(y)
    K = len(np.unique(y))
    return np.minimum(np.arange(len(y)) % (K + 1), K - 1).astype(np.int64)


@contextmanager
def _patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def _warm_ferrets(n_threads, X, y, use_streaming=False, build=None, x9=None):
    """JIT and thread pools before any timer, through every route a cell can
    take: the batch arm fits the warm-up collection on the Gram/GCV route,
    then on the LSQR route (DUAL_MAX_N patched to 8) on it and on
    ``_gate_collection``; the streaming arm fits it, and once more relabelled
    to two classes (the binary head, one weight column, has kernels the
    3-class collection does not reach). Two runs patch the schedule onto
    routes the small collection would otherwise skip: one Gram fit uses tiny
    slabs (several pieces per block) and one LSQR fit cuts 32-column
    sub-blocks. The batch runs carry unequal class sizes (``_imbalanced``),
    so the class-weighted kernels of the default head compile here; the
    first two batch runs (Gram, LSQR) are repeated on the balanced labels
    for the unweighted branch a balanced cell takes. The module constants are
    restored before the first timed fit, so the measured fits run the
    shipped configuration.

    ``x9`` (default: the warm-up collection has more than one channel)
    appends a 9-channel collection, so families of 2, 4 and 8 channels are
    compiled; a C = 1 cell keeps the univariate list. ``build`` (default
    ``_ferrets``) makes the model; the per-channel arms pass their own and
    x9=False."""
    _repo_on_path()
    import numpy as np
    import ferrets.config
    build = build or _ferrets
    if x9 is None:
        x9 = np.ndim(X) == 3 and np.shape(X)[1] > 1
    caught = (ValueError, MemoryError, RuntimeError)
    lsqr = (("DUAL_MAX_N", 8),)
    if use_streaming:
        y2 = (np.arange(len(y)) % 2).astype(np.int64)
        runs = [((), (X, y)), ((), (X, y2))]
    else:
        yi = _imbalanced(y)
        gX, gy = _gate_collection()
        runs = [((), (X, yi)), (lsqr, (X, yi)), (lsqr, (gX, _imbalanced(gy))),
                ((("GRAM_SLAB_MB", 0.01),), (X, yi)),
                (lsqr + (("FLAT_BAND_BITS", 5),), (X, yi)),
                ((), (X, y)), (lsqr, (X, y))]
    if x9:
        X9 = np.cumsum(np.random.RandomState(1).randn(24, 9, 96), axis=-1)
        y9 = (np.arange(24) % 3).astype(np.int64)
        if not use_streaming:
            y9 = _imbalanced(y9)
        runs += ([((), (X9, y9))] if use_streaming else
                 [((), (X9, y9)), (lsqr, (X9, y9))])
    for patches, (Xw, yw) in runs:
        try:
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(_patched(ferrets.config, *patch))
                build(n_threads, use_streaming).fit(Xw, yw).predict(Xw)
        except caught:
            pass


def _minirocket(n_threads, n_kernels=10000):
    from aeon.classification.convolution_based import MiniRocketClassifier
    return MiniRocketClassifier(n_kernels=n_kernels, n_jobs=n_threads,
                                random_state=RANDOM_STATE)


def _hydra(n_threads):
    import torch
    torch.set_num_threads(n_threads)
    from aeon.classification.convolution_based import HydraClassifier
    return HydraClassifier(n_jobs=n_threads, random_state=RANDOM_STATE)


def _multirocket_hydra(n_threads):
    # aeon 1.5.0's defaults spelled out: n_kernels / n_groups reach the Hydra
    # part only, the MultiRocket part keeps its own (6,250 kernels, 4
    # features per kernel).
    import torch
    torch.set_num_threads(n_threads)
    from aeon.classification.convolution_based import (
        MultiRocketHydraClassifier,
    )
    return MultiRocketHydraClassifier(n_kernels=8, n_groups=64,
                                      class_weight=None, n_jobs=n_threads,
                                      random_state=RANDOM_STATE)


def _rdst(n_threads):
    # aeon 1.5.0's defaults spelled out; None resolves inside aeon
    # (threshold_percentiles to [5, 10], shapelet_lengths to
    # [min(max(2, m // 2), 11)], estimator to StandardScaler +
    # RidgeClassifierCV with 20 alphas).
    from aeon.classification.shapelet_based import RDSTClassifier
    return RDSTClassifier(max_shapelets=10000, shapelet_lengths=None,
                          proba_normalization=0.8, threshold_percentiles=None,
                          alpha_similarity=0.5, use_prime_dilations=False,
                          estimator=None, class_weight=None,
                          n_jobs=n_threads, random_state=RANDOM_STATE)


def _quant(n_threads):
    # aeon's defaults spelled out so n_jobs can be set (aeon 1.5's default
    # ExtraTrees runs serial)
    from aeon.classification.interval_based import QUANTClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    return QUANTClassifier(
        estimator=ExtraTreesClassifier(
            n_estimators=200, max_features=0.1, criterion="entropy",
            n_jobs=n_threads, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE)


def _weasel2(n_threads):
    from aeon.classification.dictionary_based import WEASEL_V2
    return WEASEL_V2(n_jobs=n_threads, random_state=RANDOM_STATE)


def _cboss_c50(n_threads):
    from aeon.classification.dictionary_based import ContractableBOSS
    return ContractableBOSS(n_jobs=n_threads, random_state=RANDOM_STATE,
                            time_limit_in_minutes=CONTRACT_MINUTES)


def _tde_c50(n_threads):
    from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
    return TemporalDictionaryEnsemble(n_jobs=n_threads,
                                      random_state=RANDOM_STATE,
                                      time_limit_in_minutes=CONTRACT_MINUTES)


def _borf(n_threads):
    """BORF (Spinnato et al. 2024), Bag-Of-Receptive-Fields: SAX words over
    dilated, strided sliding windows, counted into a sparse bag, with a
    ridge head on the counts cast to float (experiments/borf_ridge.py)."""
    _repo_on_path()
    from experiments.borf_ridge import BORFRidge
    return BORFRidge(n_jobs=n_threads, random_state=RANDOM_STATE)


def _muse_lr(n_threads):
    # aeon 1.5 + scikit-learn 1.8: MUSE's default RidgeClassifierCV head
    # predicts one class; the logistic head is the one the benchmark runs
    from aeon.classification.dictionary_based import MUSE
    return MUSE(n_jobs=n_threads, random_state=RANDOM_STATE,
                support_probabilities=True)


def _mrsqm(n_threads):
    # single-threaded C++; no n_jobs. No Windows wheel: a separate
    # interpreter (parent.child_command, FERRETS_MRSQM_PYTHON).
    from aeon.classification.dictionary_based import MrSQMClassifier
    return MrSQMClassifier(random_state=RANDOM_STATE)


_UNI = frozenset({"ucr", "monster", "synthetic"})
_STREAM_BENCH = frozenset({"ucr", "monster", "synthetic"}) | _MV | COST

ARMS = {a.name: a for a in (
    Arm("FERRETS", _STREAM_BENCH, lambda t: _ferrets(t),
        warm=lambda t, X, y: _warm_ferrets(t, X, y)),
    Arm("FERRETS_stream", _STREAM_BENCH,
        lambda t: _ferrets(t, use_streaming=True),
        warm=lambda t, X, y: _warm_ferrets(t, X, y, use_streaming=True)),
    # the class-weight ablation: the streaming head without class weights
    Arm("FERRETS_stream_unweighted", _STREAM_BENCH,
        lambda t: _ferrets(t, use_streaming=True, stream_class_weight=None),
        warm=lambda t, X, y: _warm_ferrets(t, X, y, use_streaming=True)),
    # the channel-layout ablation: the per-channel layout
    Arm("FERRETS_perchannel", frozenset({"ucr", "synthetic"}) | _MV | COST,
        lambda t: _ferrets_perchannel(t),
        warm=lambda t, X, y: _warm_ferrets(
            t, X, y, False, build=_ferrets_perchannel, x9=False)),
    Arm("FERRETS_stream_perchannel", _STREAM_BENCH,
        lambda t: _ferrets_perchannel(t, use_streaming=True),
        warm=lambda t, X, y: _warm_ferrets(
            t, X, y, True, build=_ferrets_perchannel, x9=False)),
    Arm("MiniRocket", ALL, _minirocket),
    Arm("QUANT", ALL, _quant),
    Arm("Hydra", ALL, _hydra),
    Arm("MultiRocketHydra", frozenset({"ucr"}) | COST, _multirocket_hydra),
    Arm("RDST", frozenset({"ucr"}) | COST, _rdst),
    Arm("BORF", frozenset({"ucr"}) | _MV | COST, _borf),
    Arm("MrSQM", frozenset({"ucr", "monster"}) | COST, _mrsqm, wsl=True, slow=True,
        univariate=True),
    Arm("WEASEL2", _UNI | COST, _weasel2, slow=True, univariate=True),
    Arm("MUSE_LR", COST, _muse_lr, slow=True),
    Arm("cBOSS_c50", frozenset({"ucr", "monster"}) | COST, _cboss_c50, slow=True,
        univariate=True),
    Arm("TDE_c50", frozenset({"ucr", "monster"}) | _MV | COST, _tde_c50, slow=True),
)}


def arms_for(benchmark):
    """Arm names declared on ``benchmark``, in the default run order (cheap
    first, so competitor timings on one dataset land minutes apart)."""
    return [a.name for a in ARMS.values() if benchmark in a.benchmarks]


def build(name, n_threads):
    return ARMS[name].build(int(n_threads))


def is_slow(name):
    """Arms singled out as expensive (their fold cost on UCR: TDE 13.4 h of
    fit, cBOSS 7.9 h, WEASEL 2 1.6 h, MrSQM 0.5 h; MUSE_LR 1.5 h on UEA,
    against 0.02-0.25 h for the others). ``slow`` decides gating and run
    order only, never the folds an arm runs."""
    return ARMS[name].slow


def is_univariate(name):
    """Arms whose aeon estimator accepts one channel only: on the cost grid
    they run the C = 1 cells and the univariate anchors."""
    return ARMS[name].univariate
