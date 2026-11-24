"""The ablation of FERRETS, and the development third its defaults were
chosen on.

The ablation measures what each part of the model does and decides nothing.
It runs on these pools:

  ucr         the 112 UCR datasets minus the 20 of the memory runs
              (MEMORY_RUNS, 92), resamples 1-9
  multiverse  the Multiverse archive's core subset (65 problems) minus the
              problems MONSTER also carries (8) = 57, at the published split
              (fold 0)
  monster     the 18 MONSTER problems the Multiverse also carries
              (benchkit/scope.MONSTER_HELDOUT_PROBLEMS), fold 0, the
              streaming fit

The defaults are chosen on the development third (DEVELOPMENT_THIRD below:
at most a third of each whole archive), drawn from the same pools, length-
stratified and seeded; ``draw_development_third`` reproduces it from the
BASE rows. Loading is benchkit's (``experiments.benchkit.data.load_folds``),
so a (dataset, fold) split is the one the bench rows were measured on;
``--check-base`` re-fits BASE on a few cells and compares its accuracy with
the bench's FERRETS rows.

Arms (``ARMS``). BASE is ferrets.FERRETSClassifier with its defaults
(benchkit's FERRETS_PARAMS, ``n_jobs`` = ``--threads``); every other arm
changes the option it names, through FERRETSClassifier or, for LOGISTIC,
RIDGE_CV3, HEAVY_HALF, WORD4 and WORD2, ferrets.ablation.FERRETSAblation.

  ucr         BASE
              NO_LEVEL     level_word=False             no level word
              ZN_ONLY      word_sources=("paa",)        PAA dictionaries only
              SFA_ONLY     word_sources=("fourier",)    Fourier dictionaries
                                                        only
              NO_DIFF      use_first_difference=False   series channel only
              R50, R200    n_dictionaries=50 / 200      dictionaries per
                                                        channel
              DIL1         dilation="one"               every draw at
                                                        dilation 1
              WORD4, WORD2 word_bits=4 / 2              shorter words
              TF_COUNT     sublinear_tf=False           raw counts, not
                                                        1 + ln c
              LOGISTIC     head="logistic"              3-epoch logistic SGD
                                                        head
              STREAM_LS    use_streaming=True           streaming
                                                        least-squares head
              RIDGE_CV3    ridge_cv=3                   3-fold-CV ridge at
                                                        every n
              HEAVY_HALF   prune_frac=0.5               heaviest half, refit
              NOPRIOR      prior_shift=False            class weights, no
                                                        prior shift
              NOWEIGHT     batch_class_weight=None      prior shift, no class
                                                        weights
              NOBALANCE    prior_shift=False,           neither
                           batch_class_weight=None
              ADD_*        the optional bag transforms turned on, every
                           non-empty combination of IDF (use_idf=True), NR
                           (numerosity_reduction=True) and DF
                           (min_doc_freq=2): ADD_IDF, ADD_NR, ADD_DF,
                           ADD_IDF_NR, ADD_IDF_DF, ADD_NR_DF, ADD_ALL
              STREAM_EPOCHS_10 / _40, STREAM_ETA_HALF / _DOUBLE
                           the streaming head at 10 or 40 epochs, step 0.5
                           or 2 (ferrets.config SQ_EPOCHS, SQ_ETA0 through
                           ARM_CONFIG), paired with STREAM_LS; the choice
                           arms, run only when named, on the development
                           third (``--datasets development``)
  multiverse  BASE         (the shipped layout, cross-channel words "all")
              NO_LEVEL     as ucr
              META_NONE / META_COARSE / META_FLAGS   cross_channel_words=...
              NO_DIFF      use_first_difference=False
              NOPRIOR, NOWEIGHT, NOBALANCE, TF_COUNT, ADD_*  as ucr
  monster     BASE         the streaming fit
              NO_LEVEL     as ucr, streaming
              ADD_*        as ucr, streaming

LOGISTIC and RIDGE_CV3 have their own solvers, which refuse the batch class
weights, so they run with batch_class_weight=None (the paper pairs them with
NOWEIGHT). The cross-channel arms are identical to BASE at C = 1: such a cell
is banked ``not applicable`` rather than run.

Protocol. In-process: numba, BLAS and the estimator at ``--threads`` (set in
the environment before numpy loads); each arm warmed up once at start (Gram
route, LSQR route with DUAL_MAX_N patched to 8, both SGD schedules and a
binary fit for the streaming arms, and for Multiverse a 3- and a 9-channel
collection so every family size is compiled before any timer); per (dataset,
fold) the arms run back to back, in alternating order (reversed on odd
units), so the paired fit and predict times see the same machine. The CSV is
append-only, keyed (benchmark, dataset, fold, arm); a run skips banked keys,
so deleting rows re-runs those cells. A cell's exception is banked as its
error row. No cell cap (an in-process fit cannot be killed): a cell is
banked ``not run: ...`` when the bench CSV's FERRETS row of the same
(dataset, fold) failed, took more than ``--cap-s`` or peaked above
``--mem-cap-gb``. Before every unit a quiet-box gate waits while other
processes use more than ``--quiet-pct`` of the machine, and the run stops
without banking when another timing driver appears; ``--skip-quiet-gate`` and
``--force`` are for smoke tests.

usage:
  ablation.py run --benchmark ucr|multiverse|monster [--arms A,..]
        [--datasets D,..|development|resampled] [--limit-datasets N]
        [--folds 1-9] [--threads 10] [--csv PATH] [--dry-run]
        [--check-base [--check-cells 3]]
The released rows:
  ablation.py run --benchmark ucr                   -> results/ablation_ucr.csv
  ablation.py run --benchmark multiverse            -> results/ablation_multiverse.csv
  ablation.py run --benchmark monster               -> results/ablation_monster.csv
  ablation.py run --benchmark multiverse --arms BASE,META_NONE,META_COARSE,\\
        META_FLAGS,NO_LEVEL --datasets resampled --folds 1,2 \\
        --csv results/ablation_multiverse_resampled.csv
"""

import os
import sys

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMEXPR_MAX_THREADS")


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
        os.environ[_var] = _argv_value("--threads", "10")
    if _argv_value("--omp-wait-policy", "default") == "default":
        os.environ.pop("OMP_WAIT_POLICY", None)
    else:
        os.environ["OMP_WAIT_POLICY"] = _argv_value("--omp-wait-policy",
                                                    "default")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
BENCHMARKS = ("ucr", "multiverse", "monster")
CSV_PATH = {b: RESULTS / f"ablation_{b}.csv" for b in BENCHMARKS}
BENCH_CSV = {b: RESULTS / f"bench_{b}.csv" for b in BENCHMARKS}
FOLDS = {"ucr": tuple(range(1, 10)), "multiverse": (0,), "monster": (0,)}
N_THREADS = 10
CAP_S = 3600.0
MEM_CAP_GB = 24.0
OMP_POLICIES = ("default", "PASSIVE", "ACTIVE")
QUIET_PCT = 10.0
GATE_S = 3.0
RECHECK_S = 60.0
MAX_WAIT_H = 12.0
# the 20 UCR datasets of the memory runs (experiments/ucr_memory_runs.py),
# outside the ablation's UCR pool
MEMORY_RUNS = (
    "Adiac", "ChlorineConcentration", "CinCECGTorso", "CricketX", "ECG5000",
    "FiftyWords", "FordA", "GunPoint", "HouseTwenty", "Mallat",
    "MixedShapesRegularTrain", "Phoneme", "Rock", "SemgHandGenderCh2",
    "StarLightCurves", "SwedishLeaf", "TwoPatterns",
    "UWaveGestureLibraryAll", "Wafer", "Yoga")
# the 73 Multiverse problems of the resampled cross-channel check (folds 1-2,
# results/ablation_multiverse_resampled.csv): ``--datasets resampled``
MV_RESAMPLED_ABLATION = (
    "Alzheimers", "AppliancesEnergy_disc", "ArticularyWordRecognition",
    "BeijingPM10Quality_disc", "BeijingPM25Quality_disc", "ButtonPress",
    "DuckDuckGeese", "EmoPain", "EyesOpenShut", "FaceDetection",
    "FeedbackButton", "FeetHands", "FingerMovements", "HandMovementDirection",
    "Heartbeat", "IRDS-EFL", "IRDS-EFR", "IRDS-SAL", "IRDS-SAR", "IRDS-SFE",
    "IRDS-SFL", "IRDS-SFR", "IRDS-STL", "IRDS-STR", "ImaginedFeetHands",
    "ImaginedOpenCloseFist", "InnerSpeech", "JapaneseVowels", "KERAAL-CTK",
    "KERAAL-ELK", "KERAAL-RTK", "KIMORE-LA-C", "KIMORE-LT-C", "KIMORE-PR-C",
    "KIMORE-Sq-C", "KIMORE-TR-C", "KINECAL-3WFV", "KINECAL-GGFV",
    "KINECAL-QSEC", "KINECAL-QSEO", "Locust2022", "LongIntervalTask",
    "LowCost", "MatchingPennies", "MindReading", "MotionSenseHAR",
    "MotorImagery", "NATOPS", "OpenCloseFist", "PEMS-SF", "PhonemeSpectra",
    "PhotoStimulation", "PronouncedSpeech", "SPHERE-WUS", "ShortIntervalTask",
    "SitStand", "SongFamiliarity", "SpokenArabicDigits", "UCDHE-MP",
    "UCDHE-MP-MC", "UCDHE-Rowing", "UCDHE-Rowing-MC", "UIPRMD-DS-C",
    "UIPRMD-HS-C", "UIPRMD-IL-C", "UIPRMD-SASLR-C", "UIPRMD-SL-C",
    "UIPRMD-SSA-C", "UIPRMD-SSE-C", "UIPRMD-SSIER-C", "UIPRMD-SSS-C",
    "UIPRMD-STS-C", "VisualSpeech")

# ------------------------------------------------- the development third ---
# FERRETS' defaults are chosen on at most a third of each whole archive,
# counted in datasets at any resample. The third is drawn from the pools where
# every ablation arm has rows (dataset_list: UCR 92, Multiverse 57, MONSTER
# 18), stratified by series length, seeded; a MONSTER problem that Multiverse
# also carries splits there as MONSTER's fold 0, so a MONSTER-third problem
# with a Multiverse copy counts against Multiverse too. DEVELOPMENT_THIRD
# freezes the draw; draw_development_third(pool_lengths()) reproduces it.
DEV_SEED = 0
DEV_BANDS = (150, 700)          # series length m: m <= 150, 150 < m <= 700, m > 700
DEV_CAP = {"ucr": 37, "multiverse": 44, "monster": 9}
# the MONSTER pool problems with a Multiverse copy
MONSTER_MV_COPIES = frozenset({
    "CrowdSourced", "FordChallenge", "Opportunity", "PAMAP2", "Skoda", "STEW",
    "Tiselac", "UCIActivity", "USCActivity", "WISDM", "WISDM2"})
DEVELOPMENT_THIRD = {
    "ucr": (
        "ArrowHead", "BME", "BeetleFly", "CBF", "Chinatown", "Coffee",
        "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "ECG200", "FaceFour",
        "FacesUCR", "FreezerRegularTrain", "GunPointOldVersusYoung", "Ham",
        "Haptics", "InsectEPGRegularTrain", "InsectWingbeatSound",
        "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MoteStrain",
        "NonInvasiveFetalECGThorax2", "OSULeaf", "PigAirwayPressure", "PigCVP",
        "ProximalPhalanxOutlineAgeGroup", "RefrigerationDevices", "ScreenType",
        "SemgHandMovementCh2", "ShapeletSim", "ShapesAll",
        "SmallKitchenAppliances", "SonyAIBORobotSurface2", "ToeSegmentation2",
        "Trace", "UWaveGestureLibraryY", "Wine", "Worms"),
    "multiverse": (
        "AppliancesEnergy_disc", "AsphaltObstaclesCoordinates",
        "AsphaltRegularityCoordinates", "AustraliaRainfall_disc",
        "BeijingPM10Quality_disc", "BeijingPM25Quality_disc",
        "BoneIntensitiesAgeGroup", "BoneProbAgeGroup", "CharacterTrajectories",
        "CounterMovementJump", "EigenWorms", "EmoPain", "Epilepsy",
        "EthanolConcentration", "EyesOpenShut", "Handwriting", "Heartbeat",
        "HouseholdPowerConsumption1_disc", "IEEEPPG_disc", "IRDS-SFL",
        "KERAAL-RTK", "KIMORE-PR-C", "KINECAL-QSEO", "LSST", "Libras",
        "Locust2022", "LowCost", "MotionSenseHAR", "MotorImagery", "NATOPS",
        "PEMS-SF", "PenDigits", "PhonemeSpectra", "SelfRegulationSCP1",
        "StandWalkJump", "TactileTextureRecognition", "UCDHE-Rowing-MC",
        "UIPRMD-DS-C", "UWaveGestureLibrary"),
    "monster": (
        "CornellWhaleChallenge", "FruitFlies", "LakeIce", "PAMAP2",
        "Pedestrian", "STEW", "Skoda", "Tiselac", "UCIActivity"),
}


def length_band(m):
    """0, 1 or 2: the series-length band of m (DEV_BANDS)."""
    lo, hi = DEV_BANDS
    return 0 if m <= lo else (1 if m <= hi else 2)


def band_quotas(sizes, k):
    """k split over bands of these sizes in proportion, rounded by largest
    remainder; a tie in the remainders goes to the band of longer series."""
    tot = sum(sizes)
    exact = [s * k / tot for s in sizes]
    q = [math.floor(e) for e in exact]
    rem = [round(e - f, 12) for e, f in zip(exact, q)]
    for i in sorted(range(len(sizes)), key=lambda i: (-rem[i], -i))[:k - sum(q)]:
        q[i] += 1
    return q


def pool_lengths():
    """{archive: {dataset: m}} over the pools: the error-free BASE rows of the
    ablation's CSVs."""
    out = {}
    for b in BENCHMARKS:
        out[b] = {}
        for r in read_rows(CSV_PATH[b]):
            if r["benchmark"] == b and r["arm"] == BASE and _nan(r.get("error")):
                out[b].setdefault(r["dataset"], float(r["m"]))
    return out


def draw_development_third(lengths, copies=MONSTER_MV_COPIES, cap=DEV_CAP, seed=DEV_SEED):
    """The development third drawn from `lengths` ({archive: {dataset: m}}):
    MONSTER first, then Multiverse (its cap less the MONSTER third's
    Multiverse copies), then UCR; per archive one numpy generator seeded
    `seed`, bands short to long, each drawn without replacement from its
    names sorted. -> {archive: sorted tuple}."""
    import numpy as np
    third = {}
    for b in ("monster", "multiverse", "ucr"):
        k = cap[b] - (len(set(third["monster"]) & copies) if b == "multiverse" else 0)
        bands = [sorted(d for d, m in lengths[b].items() if length_band(m) == i) for i in range(3)]
        rng = np.random.default_rng(seed)
        picked = []
        for names, q in zip(bands, band_quotas([len(x) for x in bands], k)):
            picked += [str(x) for x in rng.choice(names, size=q, replace=False)]
        third[b] = tuple(sorted(picked))
    return {b: third[b] for b in ("ucr", "multiverse", "monster")}


def development_problems(archive):
    """The archive's problems in the development third; Multiverse also holds
    the MONSTER third's Multiverse copies (44)."""
    s = set(DEVELOPMENT_THIRD[archive])
    if archive == "multiverse":
        s |= set(DEVELOPMENT_THIRD["monster"]) & MONSTER_MV_COPIES
    return frozenset(s)


BASE = "BASE"
# the optional bag transforms, turned on: every non-empty combination
ADDITIONS = {
    "ADD_IDF": dict(use_idf=True),
    "ADD_NR": dict(numerosity_reduction=True),
    "ADD_DF": dict(min_doc_freq=2),
    "ADD_IDF_NR": dict(use_idf=True, numerosity_reduction=True),
    "ADD_IDF_DF": dict(use_idf=True, min_doc_freq=2),
    "ADD_NR_DF": dict(numerosity_reduction=True, min_doc_freq=2),
    "ADD_ALL": dict(use_idf=True, numerosity_reduction=True, min_doc_freq=2),
}
ARMS = {  # benchmark -> arm -> parameters over the defaults (BASE: none)
    "ucr": {
        "BASE": {},
        "NO_LEVEL": dict(level_word=False),
        "ZN_ONLY": dict(word_sources=("paa",)),
        "SFA_ONLY": dict(word_sources=("fourier",)),
        "NO_DIFF": dict(use_first_difference=False),
        "R50": dict(n_dictionaries=50),
        "R200": dict(n_dictionaries=200),
        "DIL1": dict(dilation="one"),
        "WORD4": dict(word_bits=4),
        "WORD2": dict(word_bits=2),
        "TF_COUNT": dict(sublinear_tf=False),
        # the logistic head and ridge_cv refuse the batch class weights
        "LOGISTIC": dict(head="logistic", batch_class_weight=None),
        "STREAM_LS": dict(use_streaming=True),
        "RIDGE_CV3": dict(ridge_cv=3, batch_class_weight=None),
        "HEAVY_HALF": dict(prune_frac=0.5),
        "NOPRIOR": dict(prior_shift=False),
        "NOWEIGHT": dict(batch_class_weight=None),
        "NOBALANCE": dict(prior_shift=False, batch_class_weight=None),
        **ADDITIONS,
        # the streaming head's step and number of epochs (ARM_CONFIG), run
        # only when named
        "STREAM_EPOCHS_10": dict(use_streaming=True),
        "STREAM_EPOCHS_40": dict(use_streaming=True),
        "STREAM_ETA_HALF": dict(use_streaming=True),
        "STREAM_ETA_DOUBLE": dict(use_streaming=True),
    },
    "multiverse": {
        "BASE": {},
        "NO_LEVEL": dict(level_word=False),
        "META_NONE": dict(cross_channel_words="none"),
        "META_COARSE": dict(cross_channel_words="coarse"),
        "META_FLAGS": dict(cross_channel_words="flags"),
        "NO_DIFF": dict(use_first_difference=False),
        "TF_COUNT": dict(sublinear_tf=False),
        "NOPRIOR": dict(prior_shift=False),
        "NOWEIGHT": dict(batch_class_weight=None),
        "NOBALANCE": dict(prior_shift=False, batch_class_weight=None),
        **ADDITIONS,
    },
    "monster": {
        "BASE": {},
        "NO_LEVEL": dict(use_streaming=True, level_word=False),
        **{a: dict(use_streaming=True, **o) for a, o in ADDITIONS.items()},
    },
}
# the arms that set a ferrets.config constant (the streaming head's settings
# are module constants, not estimator parameters): applied around the arm's
# warm-up, fit and predict, restored after; they decide a default, so they
# run only on the development third and only when named
ARM_CONFIG = {"STREAM_EPOCHS_10": {"SQ_EPOCHS": 10},
              "STREAM_EPOCHS_40": {"SQ_EPOCHS": 40},
              "STREAM_ETA_HALF": {"SQ_ETA0": 0.5},
              "STREAM_ETA_DOUBLE": {"SQ_ETA0": 2.0}}
CHOICE_ARMS = tuple(ARM_CONFIG)
LAYOUT_ARMS = ("META_NONE", "META_COARSE", "META_FLAGS")
# the bench model whose row on the same (dataset, fold) says whether an arm
# can run (the skip rule), and that --check-base compares BASE with
REFERENCE_MODEL = {"ucr": "FERRETS", "multiverse": "FERRETS",
                   "monster": "FERRETS_stream"}
# Total (fit + predict) cost of an arm relative to BASE, for the --dry-run
# estimate only: rough ratios measured with this driver at 2 threads on four
# UCR datasets and three Multiverse problems, on a shared machine (tens of
# percent of noise). Arms absent here count 1.0.
COST_FACTOR = {
    "ucr": {"ZN_ONLY": 0.77, "SFA_ONLY": 0.88, "NO_DIFF": 0.55, "R50": 0.75,
            "R200": 1.49, "DIL1": 0.38, "TF_COUNT": 1.51, "LOGISTIC": 1.63,
            "STREAM_LS": 5.31, "RIDGE_CV3": 15.56, "HEAVY_HALF": 1.94,
            "NOPRIOR": 1.28, "NOWEIGHT": 1.0, "NOBALANCE": 1.28},
    "multiverse": {"META_NONE": 0.8, "META_COARSE": 0.9, "META_FLAGS": 0.85,
                   "NO_DIFF": 0.5},
}
COLUMNS = [
    "benchmark", "dataset", "fold", "arm", "order", "n_train", "n_test",
    "n_channels", "m", "n_classes", "acc", "balanced_acc", "f1_macro",
    "f1_weighted", "time_fit", "time_pred", "n_features", "n_views",
    "n_views_drawn", "solver", "alpha", "n_threads", "bg_cpu_pct_before",
    "bg_wait_s", "bg_cpu_pct_during", "quiet_gate", "started_at", "error"]
KEY = ("benchmark", "dataset", "fold", "arm")
# the ablation class's own switches; an arm setting none of them is FERRETS
ABLATION_ONLY = ("head", "ridge_cv", "prune_frac", "word_bits")


def default_arms(benchmark):
    """The arms a run without --arms covers: every arm but the choice arms."""
    return [a for a in ARMS[benchmark] if a not in CHOICE_ARMS]


# --------------------------------------------------------------- models ---

def base_params():
    """benchkit's FERRETS parameters (the bench's FERRETS arm), so BASE
    cannot drift from the bench rows it is checked against."""
    from experiments.benchkit.arms import FERRETS_PARAMS
    return dict(FERRETS_PARAMS)


def build(benchmark, arm, n_threads):
    """The arm's estimator: ferrets.FERRETSClassifier with the arm's
    parameters over the defaults, or FERRETSAblation when the arm sets one of
    the ablation class's own switches."""
    kw = base_params()
    kw.setdefault("use_streaming", benchmark == "monster")
    opts = {} if arm == BASE else ARMS[benchmark][arm]
    kw.update(opts)
    if any(k in opts for k in ABLATION_ONLY):
        from ferrets.ablation import FERRETSAblation
        return FERRETSAblation(n_jobs=n_threads, **kw)
    import ferrets
    return ferrets.FERRETSClassifier(n_jobs=n_threads, **kw)


@contextlib.contextmanager
def patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def arm_config(arm):
    """The arm's ferrets.config patches (ARM_CONFIG) as one context; every
    constant is restored on exit, an exception included."""
    import ferrets.config
    stack = contextlib.ExitStack()
    for name, value in ARM_CONFIG.get(arm, {}).items():
        stack.enter_context(patched(ferrets.config, name, value))
    return stack


def gate_collection(seed=0):
    """A small warm-up collection that crosses the size-gated kernels
    benchkit's warm_collection (24 x C x 96, K = 3) misses: K = 6 classes,
    so the LSQR route's transposed product runs its column-range kernel, and
    one document-frequency pass over >= 2 x 65536 entries. 36 univariate
    series of 256 points whatever the benchmark: the gated kernels'
    signatures do not depend on the channel count."""
    import numpy as np
    X = np.cumsum(np.random.RandomState(seed).randn(36, 1, 256), axis=-1)
    y = (np.arange(36) % 6).astype(np.int64)
    return X, y


def warm_runs(benchmark, arm):
    """[(patch or None, (X, y)), ...]: a patch is a (ferrets constant, value)
    pair applied for that run only. The batch runs carry unequal class sizes
    (benchkit's ``_imbalanced``) so that the class-weighted kernels of the
    default head compile here, then the first Gram and LSQR runs again on
    the balanced labels for the unweighted branch."""
    import numpy as np
    from experiments.benchkit import data
    from experiments.benchkit.arms import _imbalanced
    warm = data.warm_collection(benchmark)
    if ARMS[benchmark][arm].get("use_streaming") or benchmark == "monster":
        binary = (warm[0], (np.arange(len(warm[1])) % 2).astype(np.int64))
        runs = [(None, warm), (None, binary)]
        if benchmark == "monster":     # families of 2, 4 and 8 channels
            X9 = np.cumsum(np.random.RandomState(1).randn(24, 9, 96), axis=-1)
            runs.append((None, (X9, warm[1])))
        return runs
    imb = (warm[0], _imbalanced(warm[1]))
    runs = [(None, imb)]
    if benchmark == "multiverse":      # families of 2, 4 and 8 channels
        X9 = np.cumsum(np.random.RandomState(1).randn(24, 9, 96), axis=-1)
        runs.append((None, (X9, _imbalanced(warm[1]))))
    gX, gy = gate_collection()
    runs += [(("DUAL_MAX_N", 8), imb),
             (("DUAL_MAX_N", 8), (gX, _imbalanced(gy))),
             (None, warm), (("DUAL_MAX_N", 8), warm)]
    return runs


def warm_up(benchmark, arm, n_threads):
    """JIT and thread pools once per arm before any timer. Module
    constants are restored before the first timed fit."""
    import ferrets.config
    for patch, (Xw, yw) in warm_runs(benchmark, arm):
        try:
            with arm_config(arm), (patched(ferrets.config, *patch) if patch
                                   else contextlib.nullcontext()):
                build(benchmark, arm, n_threads).fit(Xw, yw).predict(Xw)
        except (ValueError, MemoryError, RuntimeError):
            pass


# ----------------------------------------------------- datasets and refs ---

def dataset_list(benchmark):
    """The pool, smallest first (benchkit's order): the UCR datasets outside
    the memory runs, the Multiverse core problems MONSTER does not also
    carry, or the MONSTER problems the Multiverse also carries."""
    from experiments.benchkit import data
    if benchmark == "ucr":
        memory = set(MEMORY_RUNS)
        return [d for d in data.datasets("ucr") if d not in memory]
    if benchmark == "monster":
        from experiments.benchkit.scope import MONSTER_HELDOUT_PROBLEMS
        keep = set(MONSTER_HELDOUT_PROBLEMS)
        return [d for d in data.datasets("monster") if d in keep]
    core = set(data.multiverse_core())
    overlap = set(data.multiverse_overlap())
    return [d for d in data.datasets("multiverse")
            if d in core and d not in overlap]


def parse_folds(text, benchmark):
    if not text:
        return list(FOLDS[benchmark])
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_reference(benchmark):
    """{(model, dataset, fold): row dict} of the bench CSV's FERRETS rows
    (MONSTER: FERRETS_stream), which the skip rule and --check-base read."""
    import pandas as pd
    path = BENCH_CSV[benchmark]
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    df = df[df["model"] == REFERENCE_MODEL[benchmark]]
    out = {}
    for r in df.to_dict("records"):
        out[(r["model"], r["dataset"], int(r["fold"]))] = r
    return out


def _nan(v):
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v))


def skip_reason(benchmark, arm, dataset, fold, n_channels, ref, cap_s,
                mem_cap_gb=None):
    """Why a cell is banked instead of run, or None: the cross-channel arms at
    C = 1, and a banked FERRETS row that failed, took more than ``cap_s``
    seconds, or peaked above ``mem_cap_gb`` GB (this driver runs in-process,
    so a fit the shipped model needed that much memory for is not
    attempted)."""
    if arm in LAYOUT_ARMS and n_channels < 2:
        return "not applicable: C = 1 (identical to BASE)"
    model = REFERENCE_MODEL[benchmark]
    r = ref.get((model, dataset, int(fold)))
    if r is None:
        return None
    if not _nan(r.get("error")):
        return f"not run: banked {model} failed ({str(r['error'])[:90]})"
    t = sum(float(r[c]) for c in ("time_fit", "time_pred")
            if not _nan(r.get(c)))
    if t > cap_s:
        return f"not run: banked {model} took {t:.0f} s > {cap_s:.0f} s"
    if mem_cap_gb is not None:
        mb = next((float(r[c]) for c in ("mem_wset_peak_mb", "mem_peak_mb")
                   if not _nan(r.get(c))), None)
        if mb is not None and mb / 1000.0 > mem_cap_gb:
            return (f"not run: banked {model} peaked at {mb / 1000.0:.1f} GB"
                    f" > {mem_cap_gb:g} GB")
    return None


# ------------------------------------------------------------- the bank ---

class Bank:
    """Append-only CSV keyed (benchmark, dataset, fold, arm). One row is one
    line appended and fsynced, so a kill loses at most the cell in flight;
    a partial last line is closed before the next append and skipped when
    read."""

    def __init__(self, path):
        self.path = Path(path)
        self.keys = set()
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            if header != COLUMNS:
                raise SystemExit(f"{self.path}: header differs from this "
                                 "driver's COLUMNS; use another --csv")
            with open(self.path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) not in (b"\n", b"\r"):
                    f.write(b"\n")
            for r in read_rows(self.path):
                self.keys.add(self.key(r))

    @staticmethod
    def key(r):
        return (str(r["benchmark"]), str(r["dataset"]), int(r["fold"]),
                str(r["arm"]))

    def has(self, benchmark, dataset, fold, arm):
        return (benchmark, dataset, int(fold), arm) in self.keys

    def add(self, row):
        new = not self.path.exists() or not self.path.stat().st_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({c: ("" if _nan(row.get(c)) else row.get(c))
                        for c in COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        self.keys.add(self.key(row))


def read_rows(path):
    """The CSV as a list of dicts (typed columns), tolerant of a truncated
    last line; later rows replace earlier ones of the same key."""
    import io
    import pandas as pd
    if not Path(path).exists() or not Path(path).stat().st_size:
        return []
    # pandas pads a short line with NaN instead of rejecting it, so a torn
    # last line is dropped here: only lines with every field are rows
    buf = io.StringIO()
    out = csv.writer(buf, lineterminator="\n")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        out.writerow(header)
        out.writerows(r for r in reader if len(r) == len(header))
    buf.seek(0)
    df = pd.read_csv(buf, low_memory=False)
    df = df.dropna(subset=["benchmark", "dataset", "fold", "arm"])
    df = df.drop_duplicates(subset=list(KEY), keep="last")
    return df.to_dict("records")


# ---------------------------------------------------- the quiet-box gate ---

class QuietTimeout(RuntimeError):
    """The box did not become quiet within --max-wait hours."""


def own_tree():
    """This process and its children: the CPU this driver itself uses (an
    ancestor's CPU, a shell for instance, is background load)."""
    import psutil
    me = psutil.Process()
    pids = {me.pid}
    try:
        pids |= {p.pid for p in me.children(recursive=True)}
    except psutil.Error:
        pass
    return pids


def sys_cpu_snap():
    """(total, busy) CPU seconds of the machine, summed over logical CPUs."""
    import psutil
    t = psutil.cpu_times()
    tot = float(sum(t))
    return tot, tot - float(t.idle)


def busy_pct(a, b):
    dt = b[0] - a[0]
    return 100.0 * (b[1] - a[1]) / dt if dt > 0 else 0.0


def _proc_cpu():
    """{pid: (name, CPU seconds, monotonic time of that reading)}."""
    import psutil
    out = {}
    for p in psutil.process_iter(["pid", "name"]):
        if not p.info["pid"]:
            continue
        try:
            t = p.cpu_times()
        except (psutil.Error, OSError):
            continue
        out[p.info["pid"]] = (p.info["name"] or "?",
                              float(t.user + t.system), time.monotonic())
    return out


def tree_cpu_s(pids):
    """CPU seconds (user + system) of the given processes, read now."""
    import psutil
    total = 0.0
    for pid in pids:
        try:
            t = psutil.Process(pid).cpu_times()
        except (psutil.Error, OSError):
            continue
        total += float(t.user + t.system)
    return total


def sample_background(seconds=GATE_S):
    """(background %, machine busy %, top) over ``seconds``: the machine's
    busy share minus this driver's own process tree, as % of all logical
    CPUs; top = the five busiest other processes (% of the machine)."""
    import psutil
    ncpu = psutil.cpu_count() or 1
    own = own_tree()
    before = _proc_cpu()
    c0 = tree_cpu_s(own)
    s0 = sys_cpu_snap()
    t0 = time.monotonic()
    time.sleep(seconds)
    s1 = sys_cpu_snap()
    dt = time.monotonic() - t0
    c1 = tree_cpu_s(own)
    after = _proc_cpu()
    busy = busy_pct(s0, s1)
    bg = max(0.0, busy - 100.0 * (c1 - c0) / (dt * ncpu))
    top = []
    for pid, (name, s, ta) in after.items():
        if pid in own or pid not in before:
            continue
        _, s_b, t_b = before[pid]
        if ta - t_b <= 0:
            continue
        d = 100.0 * (s - s_b) / ((ta - t_b) * ncpu)
        if d >= 0.5:
            top.append((d, name, pid))
    top.sort(reverse=True)
    return bg, busy, "; ".join(f"{n}({p}) {d:.1f}%" for d, n, p in top[:5])


def quiet_gate(log, quiet_pct=QUIET_PCT, seconds=GATE_S, recheck_s=RECHECK_S,
               max_wait_h=MAX_WAIT_H, skip=False):
    """Wait until the background CPU is at most ``quiet_pct`` % of the
    machine (re-checked every ``recheck_s``); raise QuietTimeout after
    ``max_wait_h`` hours. Returns the row fields of the passing sample."""
    t_start = time.monotonic()
    while True:
        bg, busy, top = sample_background(seconds)
        waited = time.monotonic() - t_start
        if skip or bg <= quiet_pct:
            return dict(bg_cpu_pct_before=bg, bg_wait_s=waited,
                        quiet_gate="skipped" if skip else "passed")
        why = (f"background CPU {bg:.1f}% > {quiet_pct}% of the machine "
               f"(busy {busy:.1f}%); top: {top}")
        if waited >= max_wait_h * 3600:
            raise QuietTimeout(f"{why} after {waited / 3600:.1f} h of waiting")
        log(f"WAIT {why}; re-check in {recheck_s:.0f} s "
            f"(waited {waited / 60:.0f} min of {max_wait_h:g} h)")
        time.sleep(recheck_s)


# ------------------------------------------------------------- the run ----

def run_cell(benchmark, arm, fold, dataset, order, n_threads):
    """One (dataset, fold, arm): fit, predict, metrics; an exception is the
    row's ``error``."""
    import numpy as np
    import psutil
    from sklearn.metrics import balanced_accuracy_score, f1_score
    row = dict(benchmark=benchmark, dataset=dataset, fold=int(fold.fold),
               arm=arm, order=order, n_train=int(len(fold.y_train)),
               n_test=int(len(fold.y_test)),
               n_channels=int(fold.X_train.shape[1]),
               m=int(fold.X_train.shape[2]), n_classes=int(fold.n_classes),
               n_threads=n_threads, error=None,
               started_at=datetime.datetime.now().isoformat(
                   timespec="seconds"))
    gc.collect()
    model = build(benchmark, arm, n_threads)
    me = psutil.Process()
    p0 = me.cpu_times()
    s0 = sys_cpu_snap()
    c0 = timer()
    try:
        with arm_config(arm):
            t0 = timer()
            model.fit(fold.X_train, fold.y_train)
            t1 = timer()
            pred = model.predict(fold.X_test)
            t2 = timer()
        y = fold.y_test
        row.update(
            time_fit=t1 - t0, time_pred=t2 - t1,
            acc=float(np.mean(pred == y)),
            balanced_acc=float(balanced_accuracy_score(y, pred)),
            f1_macro=float(f1_score(y, pred, average="macro",
                                    zero_division=0)),
            f1_weighted=float(f1_score(y, pred, average="weighted",
                                       zero_division=0)))
    except Exception as error:  # noqa: BLE001 -- bank it, keep going
        row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    elapsed = timer() - c0
    s1 = sys_cpu_snap()
    p1 = me.cpu_times()
    own = float((p1.user + p1.system) - (p0.user + p0.system))
    busy = busy_pct(s0, s1)
    ncpu = psutil.cpu_count() or 1
    row["bg_cpu_pct_during"] = max(
        0.0, busy - 100.0 * own / (max(elapsed, 1e-9) * ncpu))
    if row["error"] is None:      # outside the timers
        a = getattr(model, "alpha_", None)
        views = getattr(model.transformer_, "views_", [])
        drawn = getattr(model, "views_all_", views)
        row.update(n_features=int(model.transformer_.n_features_),
                   n_views=int(sum(len(v) for v in views)),
                   n_views_drawn=int(sum(len(v) for v in drawn)),
                   solver=model.solver_,
                   alpha=None if a is None else float(a))
    del model
    return row


def busy_processes():
    """Other timing drivers alive (benchkit's markers; this process and its
    ancestors excluded)."""
    from experiments.benchkit.parent import busy_processes as bench_busy
    return bench_busy()


def check_environment(args):
    """The thread caps are read at import time, so the environment set before
    argparse must be the one the arguments ask for."""
    want = str(int(args.threads))
    bad = [v for v in THREAD_ENV if os.environ.get(v) != want]
    pol = os.environ.get("OMP_WAIT_POLICY") or "default"
    if bad or pol != args.omp_wait_policy:
        raise SystemExit(
            f"environment does not match the arguments: {bad or ''} "
            f"(threads {args.threads}), OMP_WAIT_POLICY {pol} vs "
            f"{args.omp_wait_policy}")


def parse_arms(args):
    allowed = ARMS[args.benchmark]
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        arms = default_arms(args.benchmark)
    bad = [a for a in arms if a not in allowed]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad} for {args.benchmark}; one "
                         f"of {tuple(allowed)}")
    if BASE not in arms:
        arms.insert(0, BASE)            # every arm is paired against it
    return list(dict.fromkeys(arms))


def plan_names(args):
    """The datasets of the run, in benchkit's order: the pool, or --datasets
    (a comma list, ``development`` for the archive's development third, or
    ``resampled`` for MV_RESAMPLED_ABLATION). Problems outside the pool go to
    their own --csv, never into the pool's CSV."""
    from experiments.benchkit import data
    pool = dataset_list(args.benchmark)
    names = pool
    if args.datasets:
        if args.datasets == "development":
            want = set(development_problems(args.benchmark))
        elif args.datasets == "resampled":
            if args.benchmark != "multiverse":
                raise SystemExit("--datasets resampled is a Multiverse list")
            want = set(MV_RESAMPLED_ABLATION)
        else:
            want = {d.strip() for d in args.datasets.split(",") if d.strip()}
        known = data.datasets(args.benchmark)
        missing = sorted(want - set(known))
        if missing:
            raise SystemExit(f"not in the {args.benchmark} archive: {missing}")
        outside = sorted(want - set(pool))
        default_csv = CSV_PATH[args.benchmark].resolve()
        if outside and (not args.csv or Path(args.csv).resolve() == default_csv):
            raise SystemExit(f"{len(outside)} dataset(s) outside the "
                             f"{args.benchmark} pool ({', '.join(outside[:5])}"
                             f"{', ...' if len(outside) > 5 else ''}) need their "
                             f"own --csv")
        names = [d for d in known if d in want]
    if args.limit_datasets:
        names = names[:int(args.limit_datasets)]
    return names


def estimate_hours(benchmark, units, arms, ref):
    """Rough hours of the planned cells: for every (dataset, fold) unit the
    banked FERRETS fit + predict time (the dataset's mean over its banked
    folds when this fold is not banked) times the arms' COST_FACTOR sum.
    Units without a banked time are counted apart."""
    model = REFERENCE_MODEL[benchmark]
    total = 0.0
    unknown = 0
    for ds, fold in units:
        r = ref.get((model, ds, int(fold)))
        if r is None:
            ts = [sum(float(x[c]) for c in ("time_fit", "time_pred")
                      if not _nan(x.get(c)))
                  for (mo, d, f), x in ref.items()
                  if mo == model and d == ds and _nan(x.get("error"))]
            t0 = sum(ts) / len(ts) if ts else None
        elif not _nan(r.get("error")):
            t0 = None
        else:
            t0 = sum(float(r[c]) for c in ("time_fit", "time_pred")
                     if not _nan(r.get(c)))
        if t0 is None:
            unknown += 1
            continue
        total += t0 * sum(COST_FACTOR.get(benchmark, {}).get(a, 1.0)
                          for a in arms)
    return total / 3600.0, unknown


def run_main(args):
    from experiments.benchkit import data
    check_environment(args)
    bench = args.benchmark
    arms = parse_arms(args)
    names = plan_names(args)
    folds = parse_folds(args.folds, bench)
    csv_path = Path(args.csv or CSV_PATH[bench])
    logf = open(args.log, "a", encoding="utf-8") if args.log else None

    def log(msg):
        line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    bank = Bank(csv_path)
    ref = load_reference(bench)
    if args.check_base:
        return check_base_main(args, bench, names, folds, ref, log)
    todo_cells = {}
    n_total = 0
    for ds in names:
        for fold in folds:
            todo = [a for a in arms if not bank.has(bench, ds, fold, a)]
            n_total += len(arms)
            if todo:
                todo_cells[(ds, fold)] = todo
    n_run = sum(len(t) for t in todo_cells.values())
    log(f"{bench}: {len(names)} datasets x folds {folds} x {len(arms)} arms "
        f"({', '.join(arms)}): {n_total - n_run} banked, {n_run} cells to "
        f"plan on {len(todo_cells)} units -> {csv_path}; {args.threads} "
        f"threads, quiet gate "
        f"{'OFF' if args.skip_quiet_gate else f'<= {args.quiet_pct}%'}")
    if args.dry_run:
        h, unknown = estimate_hours(bench, list(todo_cells), arms, ref)
        log(f"estimate: {h:.1f} h from the banked FERRETS times x "
            f"COST_FACTOR ({len(todo_cells)} units, {unknown} without a "
            f"banked time; compute only, no warm-up, loading or quiet gate)")
        log(f"busy drivers now: {busy_processes() or 'none'}")
        print(f"cells to run: {n_run}", flush=True)
        return 0
    if not args.force:
        busy = busy_processes()
        if busy:
            log(f"STOP: another timing driver is running: {busy}")
            return 3
    import numba
    numba.set_num_threads(int(args.threads))
    live = [a for a in arms
            if any(a in t for t in todo_cells.values())]
    t_w = timer()
    for arm in live:
        warm_up(bench, arm, int(args.threads))
    log(f"warm-up done ({timer() - t_w:.1f} s)")

    def gate_now():
        busy_now = [] if args.force else busy_processes()
        if busy_now:
            log(f"STOP: another timing driver is running: {busy_now}")
            return 3
        try:
            g = quiet_gate(log, args.quiet_pct, args.gate_seconds,
                           args.recheck_s, args.max_wait,
                           skip=args.skip_quiet_gate)
        except QuietTimeout as error:
            log(f"STOP: {error}")
            return 5
        return g

    unit = 0
    for ds in names:
        fold_todo = {f: todo_cells[(ds, f)] for f in folds
                     if (ds, f) in todo_cells}
        if not fold_todo:
            continue
        C = data.sizes(bench, ds)[2]
        run_folds = []
        for f, todo in fold_todo.items():
            keep = []
            for arm in todo:
                why = skip_reason(bench, arm, ds, f, C, ref, args.cap_s,
                                  args.mem_cap_gb)
                if why is None:
                    keep.append(arm)
                    continue
                n_tr, n_te, _, m = data.sizes(bench, ds)
                bank.add(dict(benchmark=bench, dataset=ds, fold=f, arm=arm,
                              n_train=n_tr, n_test=n_te, n_channels=C, m=m,
                              n_threads=args.threads, error=why,
                              started_at=datetime.datetime.now().isoformat(
                                  timespec="seconds")))
                log(f"{ds} f{f} {arm}: {why}")
            if keep:
                run_folds.append((f, keep))
        if not run_folds:
            continue
        was_cached = bench != "multiverse" or data.is_cached(ds)
        try:
            gen = data.load_folds(bench, ds, [f for f, _ in run_folds])
            for i, (f, keep) in enumerate(run_folds):
                try:
                    fold = next(gen)
                except Exception as error:  # noqa: BLE001 -- the loader's
                    # a dataset (or a resample of it) that cannot be loaded:
                    # bank this and the remaining folds' cells, move on
                    n_tr, n_te, _, m = data.sizes(bench, ds)
                    for f2, keep2 in run_folds[i:]:
                        for arm in keep2:
                            bank.add(dict(
                                benchmark=bench, dataset=ds, fold=f2, arm=arm,
                                n_train=n_tr, n_test=n_te, n_channels=C, m=m,
                                n_threads=args.threads,
                                error=f"load failed: {type(error).__name__}: "
                                      f"{str(error)[:200]}"))
                    log(f"{ds}: load failed ({type(error).__name__}: "
                        f"{error})")
                    break
                gate = gate_now()
                if isinstance(gate, int):
                    return gate
                order = keep[::-1] if unit % 2 else list(keep)
                unit += 1
                for k, arm in enumerate(order):
                    row = run_cell(bench, arm, fold, ds, k, int(args.threads))
                    busy_now = [] if args.force else busy_processes()
                    if busy_now:
                        log(f"STOP: a timing driver appeared during {ds} f{f}"
                            f" {arm}; the cell is not banked: {busy_now}")
                        return 3
                    row.update(bg_cpu_pct_before=gate["bg_cpu_pct_before"],
                               bg_wait_s=gate["bg_wait_s"],
                               quiet_gate=gate["quiet_gate"])
                    bank.add(row)
                    if row["error"]:
                        log(f"{ds} f{f} {arm}: ERROR {row['error']}")
                    else:
                        log(f"{ds} f{f} {arm}: acc {row['acc']:.4f} bal "
                            f"{row['balanced_acc']:.4f} fit "
                            f"{row['time_fit']:.3f} s pred "
                            f"{row['time_pred']:.3f} s F "
                            f"{row['n_features']} bg "
                            f"{row['bg_cpu_pct_before']:.1f}/"
                            f"{row['bg_cpu_pct_during']:.1f}%")
                del fold
                gc.collect()
        finally:
            if not was_cached and not args.keep_cache:
                # the .npz this run built; the shape .json stays (benchkit's
                # reaped state)
                data.cache_file(ds).unlink(missing_ok=True)
    log("all planned cells done")
    return 0


def check_base_main(args, bench, names, folds, ref, log):
    """Fit BASE on a few (dataset, fold) cells that have a banked FERRETS row
    and compare its metrics with that row (accuracy within ``--check-tol``,
    default 0: the same split, parameters and code give the same
    predictions). Writes nothing."""
    from experiments.benchkit import data
    import numba
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score
    numba.set_num_threads(int(args.threads))
    warm_up(bench, BASE, int(args.threads))
    model_name = REFERENCE_MODEL[bench]
    cells = [(ds, f) for ds in names for f in folds
             if (model_name, ds, f) in ref
             and _nan(ref[(model_name, ds, f)].get("error"))]
    cells = cells[:int(args.check_cells)]
    if not cells:
        log(f"check-base: no (dataset, fold) with a banked {model_name} row")
        return 4
    worst = 0.0
    for ds, f in cells:
        was_cached = bench != "multiverse" or data.is_cached(ds)
        fold = next(iter(data.load_folds(bench, ds, [f])))
        if not was_cached and not args.keep_cache:
            data.cache_file(ds).unlink(missing_ok=True)   # the one we built
        model = build(bench, BASE, int(args.threads)).fit(
            fold.X_train, fold.y_train)
        pred = model.predict(fold.X_test)
        y = fold.y_test
        got = dict(acc=float(np.mean(pred == y)),
                   balanced_acc=float(balanced_accuracy_score(y, pred)),
                   f1_macro=float(f1_score(y, pred, average="macro",
                                           zero_division=0)))
        want = ref[(model_name, ds, f)]
        diffs = {k: got[k] - float(want[k]) for k in got
                 if not _nan(want.get(k))}
        worst = max(worst, abs(diffs["acc"]))
        log(f"check-base {ds} f{f}: " + ", ".join(
            f"{k} {got[k]:.6f} vs {float(want[k]):.6f} ({diffs[k]:+.2e})"
            for k in diffs)
            + f"  [bench row {int(want['n_threads'])} threads, this run "
              f"{args.threads}]")
    ok = worst <= float(args.check_tol) + 1e-12
    log(f"check-base {'OK' if ok else 'MISMATCH'}: max |d acc| {worst:.2e} "
        f"over {len(cells)} cells (tolerance {args.check_tol})")
    return 0 if ok else 4


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    r.add_argument("--arms", default=None,
                   help="comma-separated subset of the benchmark's arms "
                        "(BASE always added): "
                        + "; ".join(f"{b}: {','.join(a)}"
                                    for b, a in ARMS.items()))
    r.add_argument("--datasets", default=None,
                   help="comma list, 'development' (the development third) or "
                        "'resampled' (Multiverse: MV_RESAMPLED_ABLATION); "
                        "default: the benchmark's pool")
    r.add_argument("--limit-datasets", type=int, default=None,
                   help="the first N planned datasets (smallest first)")
    r.add_argument("--folds", default=None,
                   help="e.g. 1-9 or 1,2,5 (default: ucr 1-9, multiverse and "
                        "monster 0)")
    r.add_argument("--threads", type=int, default=N_THREADS)
    r.add_argument("--csv", default=None,
                   help="default results/ablation_<benchmark>.csv")
    r.add_argument("--log", default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--check-base", action="store_true",
                   help="fit BASE on a few cells and compare its accuracy "
                        "with the bench CSV's FERRETS rows; no run")
    r.add_argument("--check-cells", type=int, default=3)
    r.add_argument("--check-tol", type=float, default=0.0)
    r.add_argument("--cap-s", type=float, default=CAP_S,
                   help="skip a cell whose banked FERRETS row took longer")
    r.add_argument("--mem-cap-gb", type=float, default=MEM_CAP_GB,
                   help="skip a cell whose banked FERRETS row peaked above "
                        "this (MB / 1000), it runs in this process")
    r.add_argument("--keep-cache", action="store_true",
                   help="keep the Multiverse .npz this run builds")
    r.add_argument("--quiet-pct", type=float, default=QUIET_PCT)
    r.add_argument("--gate-seconds", type=float, default=GATE_S)
    r.add_argument("--recheck-s", type=float, default=RECHECK_S)
    r.add_argument("--max-wait", type=float, default=MAX_WAIT_H,
                   help="hours to wait for a quiet box before stopping")
    r.add_argument("--skip-quiet-gate", action="store_true",
                   help="smoke runs only: sample, record, never wait")
    r.add_argument("--force", action="store_true",
                   help="smoke runs only: skip the busy check")
    r.add_argument("--omp-wait-policy", default="default",
                   choices=OMP_POLICIES)
    args = ap.parse_args(argv)
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
