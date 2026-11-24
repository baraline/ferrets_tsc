"""The memory runs: memory, model size and the one-thread cost on 20 UCR
datasets at resample 10: results/ucr_memory_runs.csv.

Per (dataset, model, n_threads) cell, in its own subprocess (the peak working
set is process-lifetime monotone on Windows; thread caps must be set before
numba, torch and BLAS import):

  acc, time_fit, time_pred     warm (JIT primed in-process on a synthetic
                               (24, 96) collection; FERRETS's with unequal
                               class sizes, so its class-weighted kernels
                               compile there)
  mem_total_mb                 peak working set over the post-warm-up base
  mem_algo_mb                  peak over the settled post-load RSS
  model_bytes                  len(pickle.dumps(fitted model))
  head_bytes                   coef_.nbytes of the model's linear head where
                               it exposes one (-1 otherwise)

Threads are capped uniformly: NUMBA_NUM_THREADS / OMP / MKL / OPENBLAS
environment variables before import, torch.set_num_threads for Hydra and
MultiRocket-Hydra, n_jobs for the estimators. Models (aeon 1.5.0,
random_state=42 unless benchkit builds them): FERRETS (benchkit's FERRETS
arm, ferrets.FERRETSClassifier defaults, 2D input), MiniRocket (10,000
kernels), QUANT, Hydra, WEASEL2, BORF, cBOSS, MultiRocketHydra and RDST
(benchkit's builders). Threads: 10 for every model, and 1 for FERRETS,
MiniRocket, Hydra, WEASEL2 and cBOSS. Datasets: the 20 of the memory runs
(experiments/ablation.MEMORY_RUNS). Fold 10 is aeon's stratified resample of
the published split seeded by 10, labels coded by rank among the training
labels. A child killed after --child-timeout seconds (default 1800) is a
"timeout after 1800s" row. Resumable: rows keyed (dataset, model, n_threads)
are skipped.

    ucr_memory_runs.py [--models A,B] [--datasets D,E] [--threads 10,1]
        [--child-timeout 1800] [--dry-run]
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
from experiments.ablation import MEMORY_RUNS  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = REPO_ROOT / "results" / "ucr_memory_runs.csv"
FOLD = 10
RANDOM_STATE = 42
DATASETS = list(MEMORY_RUNS)
ARMS = ["FERRETS", "MiniRocket", "Hydra", "WEASEL2", "cBOSS", "BORF", "QUANT",
        "MultiRocketHydra", "RDST"]
ONE_THREAD = ("FERRETS", "MiniRocket", "Hydra", "WEASEL2", "cBOSS")
TWO_D = {"FERRETS"}
THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS")
CHILD_TIMEOUT = 1800
COLUMNS = ["dataset", "model", "n_threads", "fold", "n_train", "n_test",
           "m", "n_classes", "acc", "time_fit", "time_pred",
           "mem_total_mb", "mem_algo_mb", "model_bytes", "head_bytes",
           "error"]
KEY = ("dataset", "model", "n_threads")


def build(arm, n_threads):
    if arm == "FERRETS":
        # n_jobs sets numba's count and limits BLAS for fit and predict, so
        # the 1-thread cells run one thread throughout
        from experiments.benchkit import arms as bench_arms
        return bench_arms._ferrets(n_threads)
    if arm == "MiniRocket":
        from aeon.classification.convolution_based import MiniRocketClassifier
        return MiniRocketClassifier(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm == "Hydra":
        import torch
        torch.set_num_threads(n_threads)
        from aeon.classification.convolution_based import HydraClassifier
        return HydraClassifier(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm in ("MultiRocketHydra", "RDST"):
        from experiments.benchkit.arms import _multirocket_hydra, _rdst
        return (_multirocket_hydra if arm == "MultiRocketHydra" else _rdst)(
            n_threads)
    if arm == "WEASEL2":
        from aeon.classification.dictionary_based import WEASEL_V2
        return WEASEL_V2(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm == "cBOSS":
        from aeon.classification.dictionary_based import ContractableBOSS
        return ContractableBOSS(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm == "BORF":
        from experiments.borf_ridge import BORFRidge
        return BORFRidge(n_jobs=n_threads, random_state=RANDOM_STATE)
    if arm == "QUANT":
        from experiments.ucr_family_resample10 import build_quant
        return build_quant(n_threads)
    raise ValueError(arm)


def load_fold(dataset, two_d):
    import numpy as np
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    remap = {c: i for i, c in enumerate(sorted(set(y_train)))}
    y_train = np.array([remap[v] for v in y_train], dtype=np.int64)
    y_test = np.array([remap[v] for v in y_test], dtype=np.int64)
    X_tr, y_tr, X_ts, y_ts = stratified_resample_data(
        X_train, y_train, X_test, y_test, random_state=FOLD)
    X_tr = np.ascontiguousarray(np.asarray(X_tr, dtype=np.float64))
    X_ts = np.ascontiguousarray(np.asarray(X_ts, dtype=np.float64))
    if two_d:
        X_tr, X_ts = X_tr.squeeze(1), X_ts.squeeze(1)
    return X_tr, y_tr, X_ts, y_ts, len(remap)


def child(arm, dataset, n_threads):
    import numpy as np
    import psutil

    two_d = arm in TWO_D
    warm = np.cumsum(np.random.RandomState(0).randn(24, 96), axis=1)
    warm_y = (np.arange(24) % 3).astype(np.int64)
    if arm == "FERRETS":
        # the batch head weights the classes only when their sizes differ:
        # unequal warm-up classes compile the weighted kernels here, not
        # inside the measured fit
        from experiments.benchkit.arms import _imbalanced
        warm_y = _imbalanced(warm_y).astype(np.int64)
    wx = warm if two_d else warm[:, None, :]
    build(arm, n_threads).fit(wx, warm_y).predict(wx)
    base_peak = psutil.Process().memory_info().peak_wset / 1e6

    X_tr, y_tr, X_ts, y_ts, n_classes = load_fold(dataset, two_d)
    post_load_rss = psutil.Process().memory_info().rss / 1e6
    row = dict(dataset=dataset, model=arm, n_threads=n_threads, fold=FOLD,
               n_train=len(y_tr), n_test=len(y_ts), m=X_tr.shape[-1],
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


def plan(arms, threads, datasets=DATASETS):
    """[(dataset, arm, n_threads)] in run order: arm, then thread count,
    then dataset; one thread only for the ONE_THREAD models."""
    return [(d, a, t) for a in arms for t in threads for d in datasets
            if t != 1 or a in ONE_THREAD]


def run(args):
    arms = args.models.split(",") if args.models else ARMS
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; one of {ARMS}")
    threads = [int(t) for t in args.threads.split(",")]
    datasets = DATASETS
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in DATASETS]
        if unknown:
            raise SystemExit(f"not among the memory runs' datasets: {unknown}")
        datasets = [d for d in DATASETS if d in wanted]
    every = plan(arms, threads, datasets)
    done = cellrun.banked(OUT, KEY)
    cells = [c for c in every if (c[0], c[1], str(c[2])) not in done]
    print(f"{len(every)} cells, {len(cells)} to run", flush=True)
    if args.dry_run:
        print(f"cells to run: {len(cells)}", flush=True)
        return 0
    for dataset, arm, n_threads in cells:
        env = dict(os.environ)
        for var in THREAD_ENV:
            env[var] = str(n_threads)
        rows, error = cellrun.run_child(
            [sys.executable, str(SCRIPT), "--child", arm, dataset,
             "--threads", str(n_threads)], env, args.child_timeout)
        if rows is None:
            rows = [dict(dataset=dataset, model=arm, n_threads=n_threads,
                         fold=FOLD, error=error)]
        cellrun.append_rows(OUT, rows, COLUMNS)
        got = rows[0]
        print(f"  {dataset:>24s} {arm:16s} t{n_threads:<2d} "
              f"acc={got.get('acc', float('nan')):.4f} "
              f"fit={got.get('time_fit', float('nan')):7.2f}s "
              f"pred={got.get('time_pred', float('nan')):6.2f}s "
              f"mem={got.get('mem_algo_mb', float('nan')):6.0f}MB"
              + (f"  ERROR: {got['error'][:60]}" if got.get("error") else ""),
              flush=True)
    print(f"written: {OUT}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="",
                        help="comma list (default: " + ",".join(ARMS) + ")")
    parser.add_argument("--datasets", default="",
                        help="comma-separated subset of the 20 (default all)")
    parser.add_argument("--threads", default="10,1",
                        help="comma list of thread counts (default 10,1; one "
                             "thread for " + ", ".join(ONE_THREAD) + " only)")
    parser.add_argument("--child-timeout", type=int, default=CHILD_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cells left to run; run nothing")
    parser.add_argument("--child", nargs=2, metavar=("MODEL", "DATASET"),
                        help="internal: run one cell and print a ROWS line")
    args = parser.parse_args()
    if args.child:
        child(args.child[0], args.child[1], int(args.threads))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
