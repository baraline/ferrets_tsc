"""Dataset lists, fold loaders and the synthetic benchmark.

UCR: the published train/test split (fold 0), then fold i >= 1 is aeon's
stratified resample seeded by i. Multiverse: the same protocol on all 133
problems of the Multiverse archive (Middlehurst et al. 2026, arXiv:2603.20352),
read from a float64 .npz cache that ``prefetch`` builds once per dataset; the
18 problems MONSTER also carries are included and run under the Multiverse
protocol, independently of their MONSTER cells. MONSTER: the hub's published
folds through monster_loader.load_monster_raw -- the arrays as the hub ships
them, cast to float64 and NOT normalised (the archive's own evaluation feeds
its baselines unscaled, see that function); all 29 problems. Cost: a
generated size grid (``synth.py``) plus six real anchors, each fold a repeat
of the same split -- time and memory, not accuracy. Synthetic: three tiny
generated datasets so the whole pipeline can be exercised without a
download.
"""

import functools
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import synth

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("ucr", "multiverse", "monster", "cost", "synthetic")
# Resample ids follow tsml-eval: 0 is the archive's published train/test
# split untouched, i >= 1 is aeon's stratified resample seeded by i; the cost
# benchmark reuses the fold column as a REPEAT index (3 repeats of one
# split). The default fold set of a run without --folds:
FOLDS = {"ucr": (0,), "multiverse": (0,), "monster": (0,),
         "cost": (0, 1, 2), "synthetic": (0, 1)}
MONSTER_UNIVARIATE = ("LakeIce", "InsectSound", "AudioMNIST-DS",
                      "CornellWhaleChallenge", "FruitFlies")
# (n, channels, m) of each WHOLE MONSTER dataset, read from the .npy headers
# on the hub (HfApi file sizes plus a ranged GET of each .npy header, no
# array downloaded). MONSTER's protocol is 5-fold CV over it, so a fold's
# test set is about n/5. Which arm runs on which of them is the feasibility
# gate's call (benchkit/gate.py). MONSTER_UNIVARIATE keeps the head of
# MONSTER_DATASETS; the other problems sort smallest first.
MONSTER_SHAPES = {
    "LakeIce": (129280, 1, 161), "InsectSound": (50000, 1, 600),
    "AudioMNIST-DS": (30000, 1, 4000), "CornellWhaleChallenge": (30000, 1, 4000),
    "FruitFlies": (34518, 1, 5000),
    "WISDM": (17166, 3, 100), "UCIActivity": (10299, 9, 128),
    "USCActivity": (56228, 6, 100), "FordChallenge": (36257, 30, 40),
    "CrowdSourced": (12289, 14, 256), "WISDM2": (149034, 3, 100),
    "Tiselac": (99687, 10, 23), "Skoda": (14117, 60, 100),
    "STEW": (28512, 14, 256), "Opportunity": (17386, 113, 100),
    "PAMAP2": (38856, 52, 100), "DREAMERV": (170246, 14, 256),
    "DREAMERA": (170246, 14, 256), "S2Agri-10pc-17": (5850881, 10, 24),
    "S2Agri-10pc-34": (5850881, 10, 24), "LenDB": (1244942, 3, 540),
    "TimeSen2Crop": (1135511, 9, 365), "S2Agri-17": (59268823, 10, 24),
    "S2Agri-34": (59268823, 10, 24),
    "AudioMNIST": (30000, 1, 47998), "MosquitoSound": (279566, 1, 3750),
    "WhaleSounds": (105163, 1, 2500), "Pedestrian": (189621, 1, 24),
    "Traffic": (1460968, 1, 24),
}
# Bytes monster_loader (load_monster_raw, as load_monster before it) downloads
# for one dataset at fold 0 --
# <name>_X.npy + the label file (<name>_Y.npy, else <name>_y.npy) +
# test_indices_fold_0.txt -- from the hub's own file listing (HfApi
# list_repo_tree). Exact, so the disk rule (benchkit/disk.py)
# knows what a dataset will take BEFORE it is downloaded. The arrays are
# float32, float64 or int16 depending on the dataset, so shape x 8 would
# overstate several of them.
MONSTER_HUB_BYTES = {
    "LakeIce": 84_475_497, "InsectSound": 240_458_040,
    "AudioMNIST-DS": 480_278_646, "CornellWhaleChallenge": 480_279_994,
    "FruitFlies": 1_381_035_591, "WISDM": 20_763_920,
    "UCIActivity": 47_551_988, "USCActivity": 135_475_872,
    "FordChallenge": 174_365_247, "CrowdSourced": 176_288_181,
    "WISDM2": 180_345_411, "Tiselac": 183_939_560, "Skoda": 338_935_915,
    "STEW": 409_015_182, "Opportunity": 786_012_385,
    "PAMAP2": 1_616_780_074, "DREAMERV": 2_442_193_930,
    "DREAMERA": 2_442_193_930, "S2Agri-10pc-17": 2_864_516_568,
    "S2Agri-10pc-34": 2_864_516_568, "LenDB": 8_079_340_214,
    "TimeSen2Crop": 14_932_069_702, "S2Agri-17": 29_028_408_117,
    "S2Agri-34": 29_028_408_117,
    # AudioMNIST and WhaleSounds ship float32, MosquitoSound float64, Pedestrian and
    # Traffic float32 (y is int64 everywhere)
    "AudioMNIST": 5_760_038_646, "MosquitoSound": 8_389_585_948,
    "WhaleSounds": 1_052_617_661, "Pedestrian": 20_002_032,
    "Traffic": 154_348_534,
}
# the five univariate ones first, then smallest first
MONSTER_DATASETS = MONSTER_UNIVARIATE + tuple(sorted(
    (d for d in MONSTER_SHAPES if d not in MONSTER_UNIVARIATE),
    key=lambda d: (int(np.prod(MONSTER_SHAPES[d])), d)))
# The 20 UCR datasets of the memory runs (experiments/ucr_memory_runs.py).
# They span the archive's cost range.
SCREEN_20 = ("Adiac", "ChlorineConcentration", "CinCECGTorso", "CricketX",
             "ECG5000", "FiftyWords", "FordA", "GunPoint", "HouseTwenty",
             "Mallat", "MixedShapesRegularTrain", "Phoneme", "Rock",
             "SemgHandGenderCh2", "StarLightCurves", "SwedishLeaf",
             "TwoPatterns", "UWaveGestureLibraryAll", "Wafer", "Yoga")
# name -> (n_train, n_test, C, m, K, seed)
SYNTHETIC = {"syn_small": (40, 20, 1, 64, 2, 1),
             "syn_multi": (45, 24, 3, 48, 3, 2),
             "syn_long": (30, 16, 1, 256, 2, 3)}
WARM_CHANNELS = {"ucr": 1, "monster": 1, "multiverse": 3,
                 "cost": 1, "synthetic": 1}
# The Multiverse lists (aeon's main branch -- aeon 1.5, which this benchmark
# runs on, predates them, although its tsc_zenodo already maps all 133) and
# per-dataset metadata from the archive paper's table.
MULTIVERSE_JSON = Path(__file__).with_name("multiverse.json")
# the cost benchmark's real anchors: "<benchmark>:<dataset>", each on its
# published split (MONSTER: the hub's fold 0), one small and one large per
# archive
COST_ANCHORS = ("ucr:ItalyPowerDemand", "ucr:StarLightCurves",
                "multiverse:BasicMotions", "multiverse:PEMS-SF",
                "monster:LakeIce", "monster:FruitFlies")


def _anchor(name):
    benchmark, dataset = name.split(":", 1)
    return benchmark, dataset


@dataclass
class Fold:
    fold: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int


def default_folds(benchmark, arm=None):
    """The folds a cell runs. One fold set per benchmark whatever the arm;
    ``arm`` is accepted so callers written for the slow-arm rule still work."""
    return FOLDS[benchmark]


def split_kind(benchmark, dataset, fold):
    """schema.SPLITS value of one row (see schema.SPLITS)."""
    if benchmark == "cost":
        if str(dataset).startswith("syn_"):
            return "synthetic"
        if ":" in str(dataset):  # reference to another benchmark's dataset
            return "published"
    if benchmark == "monster":
        # every MONSTER fold is one of the hub's OWN published 5-fold CV
        # splits -- there is no seeded resample for MONSTER, so unlike
        # UCR/Multiverse a fold id > 0 is still "published", not
        # "resample" (load_folds's monster_loader path never resamples).
        return "published"
    return "published" if int(fold) == 0 else "resample"


@functools.lru_cache(maxsize=None)
def _multiverse():
    return json.loads(MULTIVERSE_JSON.read_text(encoding="utf-8"))


def multiverse_names():
    """The 133 problems of the Multiverse archive, in the archive's order.
    The 18 that MONSTER also carries (``multiverse_overlap``) are included:
    each archive runs its own copy under its own protocol (published split,
    arm set, loader), so a dataset can have a row in both
    bench_multiverse.csv and bench_monster.csv."""
    return list(_multiverse()["multiverse2026"])


def multiverse_overlap():
    """The Multiverse problems MONSTER also carries (18), archive order."""
    mv = _multiverse()
    monster = set(mv["monster"])
    return [d for d in mv["multiverse2026"] if d in monster]


def multiverse_core():
    """The archive's core subset, archive order."""
    return list(_multiverse()["multiverse_core"])


@functools.lru_cache(maxsize=None)
def _ucr_info():
    return json.loads((REPO_ROOT / "data_info.json").read_text())


def data_home():
    """aeon 1.5's data home: $AEON_DATA, else ~/aeon_data. A MrSQM child
    under WSL2 is given the Windows side's home as $AEON_DATA
    (parent.wsl_command), so both sides read the same downloads."""
    env = os.environ.get("AEON_DATA")
    return Path(env) if env else Path.home() / "aeon_data"


def cache_file(name):
    """The float64 .npz one Multiverse dataset is read from; its shape is in
    the .json next to it."""
    return data_home() / "_bench_npz" / f"{name}.npz"


def is_cached(name):
    """True when a Multiverse dataset's .npz AND its shape .json are on
    disk: what the loader needs to read it without calling ``prefetch``.
    The .json alone is the normal state of a reaped dataset."""
    path = cache_file(name)
    return path.exists() and path.with_suffix(".json").exists()


def _cached_info(name):
    side = cache_file(name).with_suffix(".json")
    return json.loads(side.read_text(encoding="utf-8")) if side.exists() else None


def sizes(benchmark, dataset):
    """(n_train, n_test, n_channels, m) without loading the data -- what the
    feasibility gate and the cost order need. Multiverse prefers the cached
    shape (the _eq versions are not the archive table's longest length)."""
    if benchmark == "ucr":
        i = _ucr_info()[dataset]
        return i["training samples"], i["testing samples"], 1, i["length"]
    if benchmark == "multiverse":
        i = _cached_info(dataset)
        if i is not None:
            return i["n_train"], i["n_test"], i["n_channels"], i["m"]
        i = _multiverse()["meta"][dataset]
        return i["n_train"], i["n_test"], i["C"], i["m"]
    if benchmark == "monster":
        n, channels, m = MONSTER_SHAPES[dataset]
        return n - n // 5, n // 5, channels, m
    if benchmark == "synthetic":
        n_train, n_test, channels, m, _, _ = SYNTHETIC[dataset]
        return n_train, n_test, channels, m
    if benchmark == "cost":
        if dataset.startswith("syn_"):
            p = synth.parse_cell(dataset)
            return p["n_train"], p["n_train"] // 2, p["C"], p["m"]
        return sizes(*_anchor(dataset))
    raise ValueError(f"unknown benchmark {benchmark!r}; one of {BENCHMARKS}")


def datasets(benchmark):
    """Smallest-first (n_train x C x m) so cheap cells bank early."""
    if benchmark == "ucr":
        from aeon.datasets.tsc_datasets import univariate_equal_length
        info = _ucr_info()
        return sorted(univariate_equal_length,
                      key=lambda d: (info[d]["training samples"]
                                     * info[d]["length"], d))
    if benchmark == "multiverse":
        def cost(d):
            n_train, _, channels, m = sizes("multiverse", d)
            return (n_train * channels * m, d)
        return sorted(multiverse_names(), key=cost)
    if benchmark == "monster":
        return list(MONSTER_DATASETS)
    if benchmark == "cost":
        return synth.grid() + list(COST_ANCHORS)
    if benchmark == "synthetic":
        return list(SYNTHETIC)
    raise ValueError(f"unknown benchmark {benchmark!r}; one of {BENCHMARKS}")


def warm_channels(benchmark, dataset=None):
    """Channels of the JIT warm-up collection. MONSTER now mixes univariate
    and multivariate datasets: a multivariate one warms on 3 channels so its
    fit does not compile the multivariate code paths inside the timer."""
    if benchmark == "monster" and dataset in MONSTER_SHAPES:
        return 1 if MONSTER_SHAPES[dataset][1] == 1 else 3
    if benchmark == "cost":
        return 1 if sizes("cost", dataset)[2] == 1 else 3
    return WARM_CHANNELS[benchmark]


def warm_collection(benchmark, seed=0, n_channels=None):
    """A random-walk collection for the JIT / thread-pool warm-up."""
    C = n_channels or WARM_CHANNELS[benchmark]
    X = np.cumsum(np.random.RandomState(seed).randn(24, C, 96), axis=-1)
    y = (np.arange(24) % 3).astype(np.int64)
    return X, y


def _as3d(X):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"expected (n, channels, m), got {X.shape}")
    return X


def _encode(y_train, y_test):
    """Integer codes for both sides and the class count, the codes being the
    rank of the label among ``np.union1d(y_train, y_test)``. Fitting on the
    training labels alone would make a fold whose test side
    carries a class its training side lacks crash the child with 'y contains
    previously unseen labels' (WISDM fold 3 of the hub's subject-wise CV,
    classes 2 and 3); such a test series is just an error of the
    classifier. When the class sets agree, or the test classes are a subset
    of the training ones, the union is the training classes and the codes are
    exactly what they were."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(np.union1d(y_train, y_test))
    return (le.transform(y_train).astype(np.int64),
            le.transform(y_test).astype(np.int64), len(le.classes_))


def _synthetic_split(name):
    n_tr, n_ts, C, m, K, seed = SYNTHETIC[name]
    rng = np.random.RandomState(seed)

    def make(n):
        y = np.arange(n) % K
        t = np.linspace(0, 4 * np.pi, m)
        X = np.cumsum(rng.randn(n, C, m), axis=-1) * 0.3
        for k in range(K):  # class-dependent oscillation so acc > chance
            X[y == k] += np.sin((k + 1) * t)[None, None, :] * 2.0
        return X, y.astype(np.int64)
    X_tr, y_tr = make(n_tr)
    X_ts, y_ts = make(n_ts)
    return X_tr, y_tr, X_ts, y_ts


def _monster_split(name, fold):
    """(X_train, y_train, X_test, y_test) of one MONSTER problem on one of
    the hub's folds: float64 (n, C, m) arrays holding the values the hub
    ships, NOT normalised (monster_loader.load_monster_raw; labels as
    shipped, encoded by the caller). The arrays come back C-contiguous
    float64, so ``_as3d`` hands them on without a copy."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from monster_loader import load_monster_raw
    X_tr, y_tr, X_ts, y_ts = load_monster_raw(name, fold)
    return _as3d(X_tr), np.asarray(y_tr), _as3d(X_ts), np.asarray(y_ts)


def prefetch(name, drop_downloaded=True, loader=None):
    """Cache one Multiverse dataset as float64 .npz (+ a .json of its shape)
    and return that shape. Idempotent.

    The loader holds about 2.1x the float64 data at its peak (aeon's
    ``_load_data`` appends one (C, m) array per case to a list, then copies
    the list into one array; train is kept while test is parsed; the NaN
    check adds a bool temporary), under the feasibility gate's load floor of
    3 copies (gate.DEFAULT_COPIES), so a dataset the gate lets a cell run on
    can also be prefetched within the memory budget. Whether to call it is
    bench.py's ``cmd_prefetch`` decision (planned cells, gate, disk rule).

    aeon downloads the dataset from Zenodo when it is not local and takes
    the archive's equal-length (_eq) and imputed (_nmv) versions where they
    exist -- the versions the archive's own benchmark uses. The .ts files it
    had to DOWNLOAD are deleted once the cache is written: they are
    re-downloadable, the cache is what this benchmark reads, and the archive
    is 60 GB of text. Copies that were already local are
    left alone. The shape .json records the .ts bytes (``ts_bytes``), the
    measurement disk.MV_DOWNLOAD_FACTOR rests on.

    The .json is written before the .npz is moved into place, so a process
    killed in between leaves the state of a reaped dataset (.json, no .npz)
    and never an .npz the loader (``is_cached``) would not read."""
    path = cache_file(name)
    side = path.with_suffix(".json")
    if is_cached(name):
        return json.loads(side.read_text(encoding="utf-8"))
    home = data_home()
    bundled = False
    if loader is None:
        from aeon.datasets import load_classification
        from aeon.datasets._data_loaders import MODULE
        bundled = (Path(MODULE) / "data" / name).exists()

        def loader(split):
            return load_classification(name, split=split, load_equal_length=True,
                                       load_no_missing=True)
    folder = home / name
    was_local = bundled or folder.exists()
    X_train, y_train = loader("train")
    X_test, y_test = loader("test")
    if isinstance(X_train, list) or isinstance(X_test, list):
        raise ValueError(f"{name}: unequal-length series even in its _eq version")
    X_train, X_test = _as3d(X_train), _as3d(X_test)
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(f"{name}: train {X_train.shape} and test {X_test.shape} "
                         "do not share channels and length")
    info = dict(n_train=int(len(y_train)), n_test=int(len(y_test)),
                n_channels=int(X_train.shape[1]), m=int(X_train.shape[2]),
                n_classes=int(len(set(np.asarray(y_train).astype(str)))),
                has_nan=bool(np.isnan(X_train).any() or np.isnan(X_test).any()),
                files=(sorted(p.name for p in folder.glob("*.ts"))
                       if folder.exists() else []),
                ts_bytes=(int(sum(p.stat().st_size for p in folder.glob("*.ts")))
                          if folder.exists() else 0),
                downloaded=not was_local)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, X_train=X_train, y_train=np.asarray(y_train).astype(str),
             X_test=X_test, y_test=np.asarray(y_test).astype(str))
    side.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, path)
    if drop_downloaded and not was_local:
        shutil.rmtree(folder, ignore_errors=True)
    return info


def _published_split(benchmark, dataset):
    if benchmark == "synthetic":
        return _synthetic_split(dataset)
    if benchmark == "cost":
        if dataset.startswith("syn_"):
            return synth.synthetic_cost(**synth.parse_cell(dataset))
        inner, name = _anchor(dataset)
        if inner == "monster":
            return _monster_split(name, 0)    # the MONSTER cell's own data
        return _published_split(inner, name)
    if benchmark == "multiverse":
        path = cache_file(dataset)
        if not is_cached(dataset):
            prefetch(dataset)
        with np.load(path) as z:
            return (_as3d(z["X_train"]), z["y_train"],
                    _as3d(z["X_test"]), z["y_test"])
    from aeon.datasets import load_classification
    X_tr, y_tr = load_classification(dataset, split="train")
    X_ts, y_ts = load_classification(dataset, split="test")
    return _as3d(X_tr), y_tr, _as3d(X_ts), y_ts


def load_folds(benchmark, dataset, folds):
    """Yield one Fold at a time. The next fold's arrays are built only after
    the caller has released the previous one (the child's memory baseline
    relies on it)."""
    folds = [int(f) for f in folds]
    if benchmark == "monster":
        for fold in folds:
            X_tr, y_tr, X_ts, y_ts = _monster_split(dataset, fold)
            y_tr, y_ts, K = _encode(y_tr, y_ts)
            yield Fold(fold, X_tr, y_tr, X_ts, y_ts, K)
            # the caller has released the fold: drop this frame's references
            # too, or the next fold is built beside the previous one's arrays
            del X_tr, X_ts
        return
    if benchmark == "cost":            # repeats of ONE split, no resample
        X_tr, y_tr, X_ts, y_ts = _published_split(benchmark, dataset)
        y_tr, y_ts, K = _encode(y_tr, y_ts)
        for fold in folds:
            yield Fold(fold, _as3d(X_tr), y_tr, _as3d(X_ts), y_ts, K)
        return
    from aeon.benchmarking.resampling import stratified_resample_data
    X_tr0, y_tr0, X_ts0, y_ts0 = _published_split(benchmark, dataset)
    y_tr0, y_ts0, K = _encode(y_tr0, y_ts0)
    for fold in folds:
        if fold == 0:                      # the published split, untouched
            X_tr, y_tr, X_ts, y_ts = X_tr0, y_tr0, X_ts0, y_ts0
        else:
            X_tr, y_tr, X_ts, y_ts = stratified_resample_data(
                X_tr0, y_tr0, X_ts0, y_ts0, random_state=fold)
        yield Fold(fold, _as3d(X_tr), np.asarray(y_tr, dtype=np.int64),
                   _as3d(X_ts), np.asarray(y_ts, dtype=np.int64), K)
