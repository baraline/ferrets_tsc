"""MiniRocket on the 112 UCR datasets at resample 10:
results/ucr_minirocket_resample10.csv.

MiniRocketClassifier (aeon 1.5.0, 10,000 kernels, n_jobs=10,
random_state=42). Fold 10 is aeon's stratified resample of the published
split seeded by 10; labels are coded 0..K-1 by their rank among the sorted
training labels (int64 for aeon). One subprocess per dataset: it warms the
JIT on a synthetic (24, 96) collection, snapshots its peak working set,
loads the dataset, times fit and predict, and reports the peak over that
baseline (mem_peak_over_base_mb, data included; mem_base_mb the warm
baseline; data_mb the train and test arrays). The released rows' memory
columns come from a child that ran resamples 10-19 in turn (their time
columns are resample 10's, the first fit after the warm-up); this script's
child runs resample 10 alone. The thread count is MiniRocket's n_jobs (no
environment caps), and no cap on the child's time. Resumable: rows keyed
(dataset, model, fold) are skipped.

    ucr_minirocket_resample10.py [--datasets D,E] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cellrun  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = REPO_ROOT / "results" / "ucr_minirocket_resample10.csv"
MODEL = "MiniRocket"
FOLD = 10
N_JOBS = 10
COLUMNS = [
    "dataset", "model", "fold", "acc", "time_fit", "time_pred", "n_features",
    "n_train", "n_test", "m", "n_classes",
    "mem_peak_over_base_mb", "mem_base_mb", "data_mb", "error",
]
KEY = ("dataset", "model", "fold")


def build(n_jobs):
    from aeon.classification.convolution_based import MiniRocketClassifier
    return MiniRocketClassifier(n_jobs=n_jobs, random_state=42)


def child(dataset, n_jobs):
    import numpy as np
    import psutil
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification

    warm = np.cumsum(np.random.RandomState(0).randn(24, 96), axis=1)
    warm_y = (np.arange(24) % 3).astype(np.uint32)
    build(n_jobs).fit(warm, warm_y).predict(warm)
    base_peak = psutil.Process().memory_info().peak_wset / 1e6

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    classes = sorted(set(y_train))
    remap = {c: i for i, c in enumerate(classes)}
    y_train = np.array([remap[v] for v in y_train], dtype=np.uint32)
    y_test = np.array([remap[v] for v in y_test], dtype=np.uint32)
    data_mb = (np.asarray(X_train).nbytes + np.asarray(X_test).nbytes) / 1e6

    X_tr, y_tr, X_ts, y_ts = stratified_resample_data(
        X_train, y_train, X_test, y_test, random_state=FOLD)
    X_tr = np.asarray(X_tr)
    X_ts = np.asarray(X_ts)
    # aeon's classifiers need signed labels (uint32 gives near-chance
    # accuracy)
    y_tr = y_tr.astype(np.int64)
    y_ts = y_ts.astype(np.int64)
    model = build(n_jobs)
    row = dict(dataset=dataset, model=MODEL, fold=FOLD, n_train=len(y_tr),
               n_test=len(y_ts), m=X_tr.shape[-1], n_classes=len(classes))
    try:
        start = timer()
        model.fit(X_tr, y_tr)
        fitted = timer()
        predictions = model.predict(X_ts)
        finished = timer()
        row.update(acc=float((predictions == y_ts).mean()),
                   time_fit=fitted - start, time_pred=finished - fitted,
                   n_features=int(getattr(model, "n_features_", 0) or 0))
    except (ValueError, MemoryError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    final_peak = psutil.Process().memory_info().peak_wset / 1e6
    row.update(mem_peak_over_base_mb=final_peak - base_peak,
               mem_base_mb=base_peak, data_mb=data_mb)
    print("ROWS " + json.dumps([row]), flush=True)


def run(args):
    from aeon.datasets.tsc_datasets import univariate_equal_length
    datasets = sorted(univariate_equal_length)
    if args.datasets:
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in datasets]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}")
        datasets = [d for d in datasets if d in wanted]
    done = cellrun.banked(OUT, KEY)
    todo = [d for d in datasets if (d, MODEL, str(FOLD)) not in done]
    print(f"{len(datasets)} datasets, {len(todo)} to run", flush=True)
    if args.dry_run:
        print(f"cells to run: {len(todo)}", flush=True)
        return 0
    for dataset in todo:
        rows, error = cellrun.run_child(
            [sys.executable, str(SCRIPT), "--child", dataset,
             "--n-jobs", str(args.n_jobs)], dict(os.environ))
        if rows is None:
            rows = [dict(dataset=dataset, model=MODEL, fold=FOLD, error=error)]
        cellrun.append_rows(OUT, rows, COLUMNS)
        got = rows[0]
        print(f"  {dataset:>32s} acc={got.get('acc', float('nan')):.4f} "
              f"fit={got.get('time_fit', float('nan')):.2f}s "
              f"mem={got.get('mem_peak_over_base_mb', float('nan')):.0f}MB"
              + (f"  ERROR: {got['error'][:60]}" if got.get("error") else ""),
              flush=True)
    print(f"written: {OUT}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of the 112 (default all)")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cells left to run; run nothing")
    parser.add_argument("--child", metavar="DATASET",
                        help="internal: run one dataset and print a ROWS line")
    args = parser.parse_args()
    if args.child:
        child(args.child, args.n_jobs)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
