"""WEASEL+MUSE on the 26 equal-length UEA problems: results/uea_muse.csv.

Two configurations of aeon 1.5.0's MUSE (random_state=42, n_jobs=10): MUSE
with its default RidgeClassifierCV head, and MUSE_LR with
support_probabilities=True, which swaps in its LogisticRegression head
(with scikit-learn 1.8 the default head predicts one class: BasicMotions
scores 0.25, chance).

Per (dataset, model) cell, in its own subprocess with the thread caps
(NUMBA_NUM_THREADS / OMP / MKL / OPENBLAS = 10) set before import: a warm-up
fit on a synthetic (24, 3, 96) collection, then the published train/test
split (fold 0, no resample; labels coded by rank among the training labels),
fit and predict timed, and the same memory columns as
experiments/ucr_memory_runs.py (mem_total_mb, mem_algo_mb, model_bytes,
head_bytes). A child killed after --child-timeout seconds (default 3600, as
the released rows ran) is a "timeout after 3600s" row. Datasets run
smallest first (n_train x channels x m). Resumable: rows keyed (dataset,
model, n_threads) are skipped.

    uea_muse.py [--models MUSE,MUSE_LR] [--datasets D,E]
        [--child-timeout 3600] [--dry-run]
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from timeit import default_timer as timer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cellrun  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = REPO_ROOT / "results" / "uea_muse.csv"
FOLD = 0
RANDOM_STATE = 42
N_THREADS = 10
CHILD_TIMEOUT = 3600
ARMS = ["MUSE", "MUSE_LR"]
# aeon's multivariate_equal_length, smallest first by n_train x channels x m
DATASETS = [
    "ERing", "Libras", "AtrialFibrillation", "BasicMotions", "RacketSports",
    "Handwriting", "Epilepsy", "UWaveGestureLibrary", "PenDigits",
    "StandWalkJump", "NATOPS", "ArticularyWordRecognition", "FingerMovements",
    "LSST", "HandMovementDirection", "Cricket", "EthanolConcentration",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "Heartbeat", "PhonemeSpectra",
    "EigenWorms", "DuckDuckGeese", "PEMS-SF", "FaceDetection", "MotorImagery"]
THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS")
COLUMNS = ["dataset", "model", "n_threads", "fold", "n_train", "n_test",
           "n_channels", "m", "n_classes", "acc", "time_fit", "time_pred",
           "mem_total_mb", "mem_algo_mb", "model_bytes", "head_bytes",
           "n_pseudo_channels", "n_features", "error"]
KEY = ("dataset", "model", "n_threads")


def build(arm, n_threads):
    from aeon.classification.dictionary_based import MUSE
    if arm == "MUSE":
        return MUSE(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm == "MUSE_LR":
        return MUSE(n_jobs=n_threads, random_state=RANDOM_STATE,
                    support_probabilities=True)
    raise ValueError(arm)


def load_split(dataset):
    """The published train/test split, labels remapped to int64, 3D float64."""
    import numpy as np
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    remap = {c: i for i, c in enumerate(sorted(set(y_train)))}
    y_train = np.array([remap[v] for v in y_train], dtype=np.int64)
    y_test = np.array([remap[v] for v in y_test], dtype=np.int64)
    X_tr = np.ascontiguousarray(np.asarray(X_train, dtype=np.float64))
    X_ts = np.ascontiguousarray(np.asarray(X_test, dtype=np.float64))
    if X_tr.ndim != 3 or X_ts.ndim != 3:
        raise ValueError(f"{dataset}: expected (n, channels, m) arrays, got "
                         f"{X_tr.shape} / {X_ts.shape}")
    return X_tr, y_train, X_ts, y_test, len(remap)


def child(arm, dataset, n_threads):
    import numpy as np
    import psutil

    warm = np.cumsum(np.random.RandomState(0).randn(24, 3, 96), axis=-1)
    warm_y = (np.arange(24) % 3).astype(np.int64)
    build(arm, n_threads).fit(warm, warm_y).predict(warm)
    base_peak = psutil.Process().memory_info().peak_wset / 1e6

    X_tr, y_tr, X_ts, y_ts, n_classes = load_split(dataset)
    post_load_rss = psutil.Process().memory_info().rss / 1e6
    row = dict(dataset=dataset, model=arm, n_threads=n_threads, fold=FOLD,
               n_train=len(y_tr), n_test=len(y_ts),
               n_channels=X_tr.shape[1], m=X_tr.shape[-1],
               n_classes=n_classes)
    try:
        model = build(arm, n_threads)
        start = timer()
        model.fit(X_tr, y_tr)
        fitted = timer()
        predictions = model.predict(X_ts)
        finished = timer()
        row.update(acc=float((predictions == y_ts).mean()),
                   time_fit=fitted - start, time_pred=finished - fitted)
        try:
            row["model_bytes"] = len(pickle.dumps(
                model, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as error:  # noqa: BLE001 -- size is best-effort
            row["model_bytes"] = float("nan")
            row["error"] = f"pickle: {type(error).__name__}: {error}"[:200]
        coef = None
        for attr in ("clf_", "_estimator", "estimator_"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "coef_"):
                coef = inner.coef_
                break
        if coef is None and hasattr(model, "coef_"):
            coef = model.coef_
        row["head_bytes"] = int(coef.nbytes) if coef is not None else -1
    except (ValueError, MemoryError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    final_peak = psutil.Process().memory_info().peak_wset / 1e6
    row["mem_total_mb"] = final_peak - base_peak
    row["mem_algo_mb"] = final_peak - post_load_rss
    print("ROWS " + json.dumps([row]), flush=True)


def run(args):
    arms = args.models.split(",") if args.models else ARMS
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; one of {ARMS}")
    datasets = DATASETS
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in DATASETS]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}")
        datasets = [d for d in DATASETS if d in wanted]
    done = cellrun.banked(OUT, KEY)
    cells = [(d, a) for a in arms for d in datasets
             if (d, a, str(N_THREADS)) not in done]
    print(f"{len(arms) * len(datasets)} cells, {len(cells)} to run", flush=True)
    if args.dry_run:
        print(f"cells to run: {len(cells)}", flush=True)
        return 0
    env = dict(os.environ)
    for var in THREAD_ENV:
        env[var] = str(N_THREADS)
    for dataset, arm in cells:
        rows, error = cellrun.run_child(
            [sys.executable, str(SCRIPT), "--child", arm, dataset,
             "--threads", str(N_THREADS)], env, args.child_timeout)
        if rows is None:
            rows = [dict(dataset=dataset, model=arm, n_threads=N_THREADS,
                         fold=FOLD, error=error)]
        cellrun.append_rows(OUT, rows, COLUMNS)
        got = rows[0]
        print(f"  {dataset:>26s} {arm:8s} "
              f"acc={got.get('acc', float('nan')):.4f} "
              f"fit={got.get('time_fit', float('nan')):7.2f}s "
              f"pred={got.get('time_pred', float('nan')):6.2f}s"
              + (f"  ERROR: {got['error'][:60]}" if got.get("error") else ""),
              flush=True)
    print(f"written: {OUT}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="",
                        help="comma list (default MUSE,MUSE_LR)")
    parser.add_argument("--datasets", default="",
                        help="comma-separated subset of the 26 (default all)")
    parser.add_argument("--child-timeout", type=int, default=CHILD_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cells left to run; run nothing")
    parser.add_argument("--child", nargs=2, metavar=("MODEL", "DATASET"),
                        help="internal: run one cell and print a ROWS line")
    parser.add_argument("--threads", type=int, default=N_THREADS,
                        help="internal: the child's thread count")
    args = parser.parse_args()
    if args.child:
        child(args.child[0], args.child[1], int(args.threads))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
