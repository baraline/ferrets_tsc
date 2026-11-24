"""Seed variance of the shipped batch model on the ablation's UCR datasets.
Untimed.

Every benchmark row of FERRETS runs random_state=0 (the seed drives the
dictionary draws, the cut-point samples and, past ferrets.config.DUAL_MAX_N
training series, the LSQR route's 25% hold-out). This script refits the
shipped batch model with other seeds where the ablation measured it: the 92
UCR datasets of experiments/ablation.py (the 112 minus the 20 of the memory
runs), resamples 1-9, seeds 1-4; seed 0 is the ablation's BASE row
(results/ablation_ucr.csv). Each row also keeps the ridge alpha and the
prior shift's lambda the fit chose, beside n_train and the class count.

Unit = (dataset, fold, seed); each (dataset, fold) is loaded once and fits
all its seeds. Loading and model are the ablation's UCR BASE arm: benchkit's
data.load_folds("ucr", dataset, folds) (the published split, then aeon's
stratified resample seeded by the fold id), and
ferrets.FERRETSClassifier(**FERRETS_PARAMS, use_streaming=False,
n_jobs=--threads) with random_state=seed, fit on the training side and scored
on the test side with the ablation's metric definitions.

Columns: dataset, fold, seed, acc, balanced_acc, f1_macro, f1_weighted,
alpha (alpha_), prior_lambda (prior_lambda_), n_features, n_train, n_test,
n_classes, n_threads, error (repr of the exception; empty when the unit
ran); then solver ("gcv": alpha and lambda chosen on the leave-one-out
decisions of all n_train series; "lsqr", n_train > DUAL_MAX_N: on a 25%
hold-out drawn from the seed), prior_scores (the equal-weight score of each
lambda of ferrets.config.PRIOR_LAMBDAS, in that order, ";"-joined), m,
time_fit, time_pred (untimed run: planning only), started_at.

It does not start while a timing driver is alive (benchkit's DRIVER_MARKERS;
--force overrides). Resumable: the CSV is append-only, keyed (dataset, fold,
seed), one fsynced line per unit; a banked key is skipped, error rows
included (delete them to re-run).

usage:
  seeds_ucr.py [--datasets D,..] [--limit-datasets N] [--folds 1-9]
        [--seeds 1,2,3,4] [--threads 6] [--out results/seeds_ucr.csv]
        [--dry-run] [--force]
The released rows ran at 6 threads.
"""

import os
import sys

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMEXPR_MAX_THREADS")
N_THREADS = 6        # the released rows' thread count (untimed)


def _argv_value(flag, default):
    """``flag value`` or ``flag=value`` from sys.argv, before argparse."""
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return default


if __name__ == "__main__":
    # before numpy / numba / BLAS load; argparse rejects a bad value later
    _t = str(_argv_value("--threads", str(N_THREADS))).strip()
    if _t.isdigit() and int(_t) >= 1:
        for _var in THREAD_ENV:
            os.environ[_var] = str(int(_t))
    os.environ.pop("OMP_WAIT_POLICY", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# no numpy at its import (its heavy imports are inside its functions)
from experiments import ablation as A  # noqa: E402

BENCH = "ucr"
OUT_DEFAULT = REPO_ROOT / "results" / "seeds_ucr.csv"
BASE_CSV = A.CSV_PATH[BENCH]          # results/ablation_ucr.csv
SEEDS = (1, 2, 3, 4)
COLUMNS = [
    "dataset", "fold", "seed", "acc", "balanced_acc", "f1_macro",
    "f1_weighted", "alpha", "prior_lambda", "n_features", "n_train",
    "n_test", "n_classes", "n_threads", "error",
    "solver", "prior_scores", "m", "time_fit", "time_pred", "started_at"]
EXIT_BUSY = 3


def log(msg):
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def _stamp():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _blank(v):
    return v is None or (isinstance(v, str) and not v.strip()) or (
        isinstance(v, float) and math.isnan(v))


def parse_ints(text, default, what):
    """A comma list of ints >= 0, ranges a-b allowed; ``default`` when
    empty."""
    if text is None or not str(text).strip():
        return sorted(set(int(v) for v in default))
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part[1:]:
                a, b = part.split("-", 1)
                out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        except ValueError:
            raise SystemExit(f"--{what}: not a list of ints: {text!r}")
    if not out or min(out) < 0:
        raise SystemExit(f"--{what}: give ints >= 0, got {text!r}")
    return sorted(set(out))


def _fmt_ints(vals):
    return ",".join(str(v) for v in vals)


# --------------------------------------------------------------- model ----

def model_params(seed):
    """The ablation's BASE parameters (ablation.build("ucr", "BASE", .):
    benchkit's FERRETS_PARAMS, batch fit) with the seed."""
    kw = A.base_params()
    kw.setdefault("use_streaming", False)
    kw["random_state"] = int(seed)
    return kw


def build(seed, n_threads):
    import ferrets
    return ferrets.FERRETSClassifier(n_jobs=int(n_threads),
                                     **model_params(seed))


def check_model_matches_base(n_threads):
    """Seed 0 must be the ablation's BASE estimator, parameter for
    parameter (BASE's rows are this script's seed 0)."""
    want = A.build(BENCH, A.BASE, int(n_threads)).get_params()
    got = build(0, n_threads).get_params()
    if got != want:
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                if got.get(k) != want.get(k)}
        raise SystemExit(f"seed-0 model differs from the ablation's BASE: {diff}")


def metrics(y, pred):
    """The ablation's four metrics: y the test labels, pred the
    predictions."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return dict(
        acc=float(np.mean(pred == y)),
        balanced_acc=float(balanced_accuracy_score(y, pred)),
        f1_macro=float(f1_score(y, pred, average="macro",
                                zero_division=0)),
        f1_weighted=float(f1_score(y, pred, average="weighted",
                                   zero_division=0)))


def fitted(model):
    """What a fit chose, read after predict (outside the timers)."""
    a = getattr(model, "alpha_", None)
    lam = getattr(model, "prior_lambda_", None)
    sc = getattr(model, "prior_scores_", None)
    return dict(
        alpha=None if a is None else float(a),
        prior_lambda=None if lam is None else float(lam),
        prior_scores=(None if sc is None
                      else ";".join(repr(float(v)) for v in sc)),
        n_features=int(model.transformer_.n_features_),
        solver=getattr(model, "solver_", None))


def run_unit(dataset, fold, seed, n_threads):
    """One (dataset, fold, seed): fit, predict, metrics; an exception is
    printed and banked as the row's ``error`` (its repr)."""
    row = dict(dataset=dataset, fold=int(fold.fold), seed=int(seed),
               n_train=int(len(fold.y_train)), n_test=int(len(fold.y_test)),
               n_classes=int(fold.n_classes), m=int(fold.X_train.shape[2]),
               n_threads=int(n_threads), error=None, started_at=_stamp())
    model = None
    try:
        model = build(seed, n_threads)
        t0 = timer()
        model.fit(fold.X_train, fold.y_train)
        t1 = timer()
        pred = model.predict(fold.X_test)
        t2 = timer()
        row.update(metrics(fold.y_test, pred))
        row.update(time_fit=t1 - t0, time_pred=t2 - t1, **fitted(model))
    except Exception as error:  # noqa: BLE001 -- bank it, keep going
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        row["error"] = repr(error)[:2000]
        model = None
        gc.collect()
    del model
    return row


# ------------------------------------------------------------- the bank ---

class Bank:
    """Append-only CSV keyed (dataset, fold, seed). One unit is one line
    appended and fsynced, so a kill loses at most the unit in flight; a torn
    last line is skipped when read and closed before the next append."""

    def __init__(self, path):
        self.path = Path(path)
        self.keys = set()
        self._closed_tail = False
        if not (self.path.exists() and self.path.stat().st_size):
            return
        with open(self.path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != COLUMNS:
                raise SystemExit(f"{self.path}: header differs from this "
                                 "script's COLUMNS; use another --out")
            for rec in reader:
                if len(rec) != len(header):
                    continue                    # a torn line
                r = dict(zip(header, rec))
                try:
                    self.keys.add((r["dataset"], int(r["fold"]), int(r["seed"])))
                except ValueError:
                    continue

    def has(self, dataset, fold, seed):
        return (dataset, int(fold), int(seed)) in self.keys

    def add(self, row):
        new = not self.path.exists() or not self.path.stat().st_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not new and not self._closed_tail:
            with open(self.path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) not in (b"\n", b"\r"):
                    f.write(b"\n")
        self._closed_tail = True
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({c: ("" if _blank(row.get(c)) else row.get(c))
                        for c in COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        self.keys.add((str(row["dataset"]), int(row["fold"]), int(row["seed"])))


def load_base_rows(path=BASE_CSV):
    """{(dataset, fold): row} of the ablation's UCR BASE arm, as text; a
    later row replaces an earlier one of the same key and a torn line is
    skipped."""
    out = {}
    p = Path(path)
    if not (p.exists() and p.stat().st_size):
        return out
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        for rec in reader:
            if len(rec) != len(header):
                continue
            r = dict(zip(header, rec))
            if r.get("benchmark") != BENCH or r.get("arm") != A.BASE:
                continue
            try:
                out[(r["dataset"], int(r["fold"]))] = r
            except (KeyError, ValueError):
                continue
    return out


def _base_ok(base, dataset, fold):
    r = base.get((dataset, int(fold)))
    return r if r is not None and _blank(r.get("error")) else None


# ------------------------------------------------------------- planning ---

def plan_names(args):
    """The ablation's UCR datasets, smallest first, cut to --datasets /
    --limit-datasets."""
    names = A.dataset_list(BENCH)
    if args.datasets:
        want = [d.strip() for d in args.datasets.split(",") if d.strip()]
        missing = [d for d in want if d not in names]
        if missing:
            raise SystemExit(f"not among the ablation's UCR datasets: {missing}")
        keep = set(want)
        names = [d for d in names if d in keep]
    if args.limit_datasets:
        names = names[:int(args.limit_datasets)]
    return names


def check_environment(n_threads):
    """The thread caps are read when numpy / numba load, so the environment
    set from argv must be the one --threads asks for."""
    want = str(int(n_threads))
    bad = sorted(v for v in THREAD_ENV if os.environ.get(v) != want)
    if bad:
        raise SystemExit(f"thread environment {bad} is not --threads "
                         f"{n_threads}: run this file as a script (it sets "
                         "them from argv before numpy loads)")


def _base_time(base, dataset, fold):
    """The banked BASE fit + predict seconds of (dataset, fold), else the
    dataset's mean over its banked folds, else None."""
    def t(r):
        try:
            return float(r["time_fit"]) + float(r["time_pred"])
        except (KeyError, TypeError, ValueError):
            return None
    r = _base_ok(base, dataset, fold)
    if r is not None and t(r) is not None:
        return t(r)
    ts = [t(r) for (d, _), r in base.items()
          if d == dataset and _blank(r.get("error")) and t(r) is not None]
    return sum(ts) / len(ts) if ts else None


def _split_differs(fold, base_row):
    """[(column, ours, banked)] where the loaded fold disagrees with its
    banked BASE row (the same split must give the same sizes)."""
    ours = dict(n_train=int(len(fold.y_train)), n_test=int(len(fold.y_test)),
                n_classes=int(fold.n_classes), m=int(fold.X_train.shape[2]))
    out = []
    for k, v in ours.items():
        try:
            b = int(float(base_row[k]))
        except (KeyError, TypeError, ValueError):
            continue
        if b != v:
            out.append((k, v, b))
    return out


def _g(v, spec="g"):
    try:
        return "-" if _blank(v) else format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _unit_line(row):
    if not _blank(row.get("error")):
        return f"ERROR {row['error']}"
    return (f"acc {_g(row.get('acc'), '.4f')} bal "
            f"{_g(row.get('balanced_acc'), '.4f')} alpha "
            f"{_g(row.get('alpha'))} lambda {_g(row.get('prior_lambda'))} F "
            f"{_g(row.get('n_features'))} {row.get('solver')}")


# ------------------------------------------------------------- the run ----

def run_main(args):
    from experiments.benchkit import data
    T = int(args.threads)
    names = plan_names(args)
    folds = parse_ints(args.folds, A.FOLDS[BENCH], "folds")
    seeds = parse_ints(args.seeds, SEEDS, "seeds")
    out = Path(args.out)
    bank = Bank(out)
    base = load_base_rows()
    plan = [(d, f, s) for d in names for f in folds for s in seeds]
    todo_by_ds = {}
    for d, f, s in plan:
        if not bank.has(d, f, s):
            todo_by_ds.setdefault(d, []).append((f, s))
    n_todo = sum(len(v) for v in todo_by_ds.values())
    log(f"seeds_ucr: {len(names)} datasets x folds {_fmt_ints(folds)} x "
        f"seeds {_fmt_ints(seeds)} = {len(plan)} units: "
        f"{len(plan) - n_todo} banked, {n_todo} to run -> {out}; {T} threads")
    if args.dry_run:
        est = sum(_base_time(base, d, f) or 0.0
                  for d, todo in todo_by_ds.items() for f, _ in todo)
        log(f"estimate: ~{est / 60:.1f} min of fit + predict at 10 threads "
            f"(the banked BASE times, once per unit), plus loading")
        print(f"cells to run: {n_todo}", flush=True)
        return 0
    check_environment(T)
    check_model_matches_base(T)
    if not args.force:
        busy = A.busy_processes()
        if busy:
            log(f"STOP: a timing driver is running (--force overrides): {busy}")
            return EXIT_BUSY
    import numba
    numba.set_num_threads(T)
    t_start = timer()
    done = n_err = 0
    for d in names:
        todo = todo_by_ds.get(d)
        if not todo:
            continue
        fold_todo = {}
        for f, s in todo:
            fold_todo.setdefault(f, []).append(s)
        fold_list = sorted(fold_todo)
        n_tr, n_te, _, m = data.sizes(BENCH, d)
        log(f"== {d} (n_train {n_tr}, n_test {n_te}, m {m}): {len(todo)} units")
        gen = data.load_folds(BENCH, d, fold_list)
        try:
            for i, f in enumerate(fold_list):
                try:
                    fold = next(gen)
                    if int(fold.fold) != f:
                        raise RuntimeError(f"loader yielded fold {fold.fold},"
                                           f" expected {f}")
                except Exception as error:  # noqa: BLE001 -- the loader's
                    traceback.print_exc(file=sys.stdout)
                    sys.stdout.flush()
                    # bank this and the dataset's remaining units, move on
                    left = [(f2, s) for f2 in fold_list[i:]
                            for s in fold_todo[f2]]
                    for f2, s in left:
                        bank.add(dict(dataset=d, fold=f2, seed=s,
                                      n_train=n_tr, n_test=n_te, m=m,
                                      n_threads=T, started_at=_stamp(),
                                      error=f"load failed: {error!r}"[:2000]))
                    done += len(left)
                    n_err += len(left)
                    log(f"{d}: load failed at fold {f} ({error!r}); "
                        f"{len(left)} units banked as errors")
                    break
                b = _base_ok(base, d, f)
                bad = _split_differs(fold, b) if b is not None else []
                if bad:
                    log(f"WARNING {d} f{f}: split differs from its BASE row "
                        f"(column, here, banked): {bad}")
                for s in fold_todo[f]:
                    row = run_unit(d, fold, s, T)
                    bank.add(row)
                    done += 1
                    n_err += not _blank(row.get("error"))
                    log(f"[{done}/{n_todo}] {d} f{f} s{s}: {_unit_line(row)}")
                del fold
                gc.collect()
        finally:
            gen.close()
    log(f"run over: {done} units in {timer() - t_start:.1f} s, {n_err} "
        f"error rows written (delete them to re-run)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=None,
                    help="comma list (default: the ablation's 92 UCR datasets)")
    ap.add_argument("--limit-datasets", type=int, default=None,
                    help="the first N planned datasets (smallest first)")
    ap.add_argument("--folds", default=None,
                    help="e.g. 1-9 or 1,2,5 (default 1-9)")
    ap.add_argument("--seeds", default=None,
                    help="e.g. 1-4 (default 1,2,3,4)")
    ap.add_argument("--threads", type=int, default=N_THREADS,
                    help=f"numba / BLAS threads and n_jobs (default {N_THREADS})")
    ap.add_argument("--out", default=str(OUT_DEFAULT),
                    help="output CSV (default results/seeds_ucr.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    ap.add_argument("--force", action="store_true",
                    help="start even if a timing driver is running")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("--threads must be >= 1")
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
