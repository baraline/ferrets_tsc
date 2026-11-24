"""The dictionary family, Hydra and QUANT on the 112 UCR datasets at
resample 10: results/ucr_family_resample10.csv.

Models (aeon 1.5.0, random_state=42, n_jobs=10): WEASEL2 (WEASEL_V2), Hydra
(HydraClassifier, torch threads 10), MrSQM (MrSQMClassifier, aeon defaults;
single-threaded C++, no Windows wheel: it runs in the FERRETS_MRSQM_PYTHON
interpreter inside WSL, see benchkit/parent.py), TDE
(TemporalDictionaryEnsemble), BORF (experiments/borf_ridge.py), cBOSS
(ContractableBOSS) and QUANT (QUANTClassifier with aeon's default ExtraTrees,
n_jobs set). Fold 10 is aeon's stratified resample of the published split
seeded by 10, labels encoded on the training labels.

One subprocess per (dataset, model) cell with the thread caps
(NUMBA_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS =
10) set before numpy, numba, torch and BLAS load; the child fits and predicts
once on SmoothSubspace (JIT and thread pools primed), then times fit and
predict on the cell and prints one ROWS line. No cap by default: the
released rows mix uncapped cells (n_threads empty: measured in-process, n_jobs
10, no cap) and cells under a 3600 s cap on the whole child (a "timeout after
3600s" row is the cell's entry); ``--cap`` applies that cap. Models run in
the order given, datasets smallest first (training samples x length).
Resumable: rows are keyed (dataset, model, fold) and appended; delete rows to
re-run those cells.

    ucr_family_resample10.py [--models WEASEL2,Hydra,...] [--datasets D,E]
        [--cap] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

N_THREADS = 10
THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS")
if "--child" in sys.argv:
    # BLAS reads its cap at numpy import and numba at its own import
    for _var in THREAD_ENV:
        os.environ[_var] = str(N_THREADS)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cellrun  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = REPO_ROOT / "results" / "ucr_family_resample10.csv"
FOLD = 10
N_JOBS = N_THREADS
RANDOM_STATE = 42
CAP_S = 3600
MODELS = ["Hydra", "cBOSS", "TDE", "WEASEL2", "QUANT", "BORF", "MrSQM"]
COLUMNS = ["dataset", "model", "fold", "n_train", "n_test", "m",
           "n_classes", "acc", "time_fit", "time_pred", "n_threads",
           "error"]
KEY = ("dataset", "model", "fold")


def build_quant(n_jobs):
    """aeon QUANT at its defaults (interval depth 6, 200 extremely
    randomized trees, max_features 0.1, entropy), spelled out so its n_jobs
    can be set (the default estimator runs serial)."""
    from aeon.classification.interval_based import QUANTClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    return QUANTClassifier(
        estimator=ExtraTreesClassifier(
            n_estimators=200, max_features=0.1, criterion="entropy",
            n_jobs=n_jobs, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE)


def build(name):
    if name == "Hydra":
        import torch
        torch.set_num_threads(N_THREADS)
        from aeon.classification.convolution_based import HydraClassifier
        return HydraClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE)
    if name == "cBOSS":
        from aeon.classification.dictionary_based import ContractableBOSS
        return ContractableBOSS(n_jobs=N_JOBS, random_state=RANDOM_STATE)
    if name == "TDE":
        from aeon.classification.dictionary_based import (
            TemporalDictionaryEnsemble,
        )
        return TemporalDictionaryEnsemble(n_jobs=N_JOBS,
                                          random_state=RANDOM_STATE)
    if name == "WEASEL2":
        from aeon.classification.dictionary_based import WEASEL_V2
        return WEASEL_V2(n_jobs=N_JOBS, random_state=RANDOM_STATE)
    if name == "QUANT":
        return build_quant(N_JOBS)
    if name == "BORF":
        from experiments.borf_ridge import BORFRidge
        return BORFRidge(n_jobs=N_JOBS, random_state=RANDOM_STATE)
    if name == "MrSQM":
        from aeon.classification.dictionary_based import MrSQMClassifier
        return MrSQMClassifier(random_state=RANDOM_STATE)
    raise ValueError(name)


def load_fold(dataset):
    import numpy as np
    from aeon.benchmarking.resampling import stratified_resample_data
    from aeon.datasets import load_classification
    from sklearn.preprocessing import LabelEncoder

    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train).astype(np.int64)
    y_test = le.transform(y_test).astype(np.int64)
    return stratified_resample_data(X_train, y_train, X_test, y_test,
                                    random_state=FOLD)


def child(name, dataset):
    """One cell: warm up on SmoothSubspace, then time fit / predict. Prints
    a single ``ROWS <json>`` line; never writes the CSV."""
    X_tr, y_tr, X_ts, y_ts = load_fold("SmoothSubspace")
    build(name).fit(X_tr, y_tr).predict(X_ts)  # warm JIT / thread pools

    X_tr, y_tr, X_ts, y_ts = load_fold(dataset)
    row = dict(dataset=dataset, model=name, fold=FOLD,
               n_train=len(y_tr), n_test=len(y_ts),
               m=X_tr.shape[-1], n_classes=len(set(y_tr)),
               n_threads=N_THREADS)
    try:
        model = build(name)
        start = timer()
        model.fit(X_tr, y_tr)
        fitted = timer()
        acc = float((model.predict(X_ts) == y_ts).mean())
        finished = timer()
        row.update(acc=acc, time_fit=fitted - start,
                   time_pred=finished - fitted)
    except (ValueError, MemoryError, RuntimeError) as error:
        row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    print("ROWS " + json.dumps([row]), flush=True)


def child_command(name, dataset, cap=False):
    """(argv, timeout) of one cell's child: this script with ``--child``
    (MrSQM through WSL on Windows), and CAP_S seconds with ``cap``, else no
    timeout."""
    args = ["--child", name, dataset]
    if name == "MrSQM" and sys.platform == "win32":
        from experiments.benchkit.parent import wsl_command
        argv = wsl_command(SCRIPT.name, args, N_THREADS)
    else:
        argv = [sys.executable, str(SCRIPT)] + args
    return argv, (CAP_S if cap else None)


def ordered_datasets(wanted=None):
    from aeon.datasets.tsc_datasets import univariate_equal_length
    with open(REPO_ROOT / "data_info.json") as f:
        info = json.load(f)
    names = sorted(univariate_equal_length,
                   key=lambda d: info[d]["training samples"] * info[d]["length"])
    if wanted:
        unknown = [d for d in wanted if d not in names]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}")
        names = [d for d in names if d in wanted]
    return names


def run(args):
    from experiments.benchkit.parent import interpreter_missing
    models = args.models.split(",") if args.models else MODELS
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; one of {MODELS}")
    datasets = ordered_datasets([d.strip() for d in args.datasets.split(",")
                                 if d.strip()] if args.datasets else None)
    done = cellrun.banked(OUT, KEY)
    cells = [(name, d) for name in models for d in datasets
             if (d, name, str(FOLD)) not in done]
    cap = CAP_S if args.cap else None
    print(f"{len(models) * len(datasets)} cells, {len(cells)} to run, "
          f"{N_THREADS} threads, " + (f"{cap} s per cell" if cap else "no cap"),
          flush=True)
    if args.dry_run:
        print(f"cells to run: {len(cells)}", flush=True)
        return 0
    env = dict(os.environ)
    for var in THREAD_ENV:
        env[var] = str(N_THREADS)
    left = 0
    for name, dataset in cells:
        if name == "MrSQM" and interpreter_missing("MrSQM"):
            print(f"  {name} {dataset}: {interpreter_missing('MrSQM')}; not "
                  f"banked", flush=True)
            left += 1
            continue
        argv, timeout = child_command(name, dataset, cap=args.cap)
        rows, error = cellrun.run_child(argv, env, timeout)
        if rows is None:
            rows = [dict(dataset=dataset, model=name, fold=FOLD,
                         n_threads=N_THREADS, error=error)]
        cellrun.append_rows(OUT, rows, COLUMNS)
        got = rows[0]
        print(f"  {name:7s} {dataset:>28s} "
              f"acc={got.get('acc', float('nan')):.4f} "
              f"fit={got.get('time_fit', float('nan')):8.2f}s "
              f"pred={got.get('time_pred', float('nan')):7.2f}s"
              + (f"  ERROR: {got['error'][:60]}" if got.get("error") else ""),
              flush=True)
    print(f"written: {OUT}" + (f"; {left} cells left unbanked" if left else ""))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=None,
                        help="comma-separated subset of " + ",".join(MODELS))
    parser.add_argument("--datasets", default=None,
                        help="comma-separated subset of the 112 (default all)")
    parser.add_argument("--cap", action="store_true",
                        help=f"kill a cell's child after {CAP_S} s and bank a "
                             f"timeout row (default: no cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cells left to run; run nothing")
    parser.add_argument("--child", nargs=2, metavar=("MODEL", "DATASET"),
                        help="internal: run one cell and print a ROWS line")
    args = parser.parse_args()
    if args.child:
        child(args.child[0], args.child[1])
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
