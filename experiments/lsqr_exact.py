"""The batch head's iterative ridge route against the exact ridge, where both
are feasible (3,000 < n_train <= 10,000). Untimed.

Up to ferrets.config.DUAL_MAX_N (3000) training series the batch head solves
the ridge exactly on the dual (classification/gram.py, solver_ "gcv"): the
n x n Gram of the centred design, its eigendecomposition, and the exact
leave-one-out squared error of every alpha of the grid; the prior shift's
lambda is chosen on the exact leave-one-out decisions of the chosen alpha.
Above it the head takes the LSQR route (classification/lsqr.py,
kernels/krylov.py, solver_ "lsqr"):

  * a stratified 25% hold-out (config.HOLDOUT, seeded by seed_); on the 75%
    every alpha of the grid is solved at once per class column (one
    Golub-Kahan bidiagonalisation of the centred design, fully
    re-orthogonalised, each alpha solved on the projected problem, LSQR's
    two stopping tests with atol = btol = config.RIDGE_TOL (1e-2), the column
    stopping when all its alphas pass or at config.RIDGE_MAX_ITER (200)
    steps; the classes run in lockstep groups);
  * alpha = the first alpha of highest hold-out accuracy (cv_score_);
  * the model is refitted on all rows at that alpha (one more solve per
    class, the same tests and cap); coef_ is that solve (float32);
  * the prior shift's lambda is chosen on the hold-out decisions of the
    chosen alpha and folded into intercept_.

This script fits four routes on every problem where the exact route is still
affordable:

  lsqr_default  the shipped model (benchkit's FERRETS_PARAMS through the
                FERRETS arm's builder, batch head): the LSQR route
  dual_forced   the same model with ferrets.config.DUAL_MAX_N set to
                --dual-max-n (10000) for that fit only (patched on
                ferrets.config, restored after): the exact dual route,
                alpha and lambda from exact leave-one-out decisions
  lsqr_fixed    the LSQR route at one alpha, lsqr_default's alpha_, with
                prior_shift=False: no hold-out, the one solve on all rows
  exact_fixed   the dual route at that alpha, prior_shift=False

lsqr_fixed and exact_fixed differ only in the solver: same design, same
alpha, no shift. Their coefficient distance is the LSQR solve's error;
lsqr_default against dual_forced is the shipped route against the exact one
with each route's own alpha and lambda rule. Every route is built by the same
builder with the same class weights, so the four routes solve the same
(weighted) ridge.

Problems: UCR FordA, FordB, Crop, ElectricDevices (n_train 3601-8926) at
folds 0, 1, 2 (fold 0 the published split, fold >= 1 aeon's stratified
resample seeded by the fold), and the 13 Multiverse problems with 3000 <
n_train <= 10000 at fold 0 (MV_SET), loaded by benchkit's loader
(experiments.benchkit.data.load_folds). Units run smallest n_train first.

Iteration counts. The lsqr module's functions are wrapped in that module for
the duration of each fit (the package is not edited): _lsqr_holdout /
_lsqr_solve and their weighted twins _lsqr_holdout_ops / _lsqr_solve_ops
name the phase ("holdout": the multi-alpha solve on the 75%; "final": the
solve on all rows), _lsqr_group sees one lockstep group (its operator is
wrapped to log how many columns every forward product carries, which gives
the exact multiset of the columns' Golub-Kahan step counts), and _gk_store
counts the steps of each column that continues. A class solve's count is its
number of Golub-Kahan steps; it "stopped on the cap" when that count reached
RIDGE_MAX_ITER. The wrappers pass every array through unchanged, so the fit
is the shipped one bit for bit.

Memory. The dual route holds the n x n float64 Gram (800 MB at n = 10,000)
and scipy's eigh workspace: at most about 4 n^2 x 8 bytes (3.2 GB at
10,000) beside the encoded design; the largest problem (CrowdSourced) peaks
near 10.5 GB.

Columns (one row per route, four rows per unit (benchmark, dataset, fold)):
  benchmark, dataset, fold, route
  n_train, n_test, n_classes, n_features
  alpha, prior_lambda    alpha_ and prior_lambda_ of the fit
  iters_max, iters_mean  Golub-Kahan steps over the fit's class solves
                         (lsqr_default: the hold-out solves and the final
                         solves; empty for the dual routes)
  cap_hits               class solves stopped on the cap (RIDGE_MAX_ITER)
  n_class_solves         lockstep class columns solved
  acc, bacc, f1m         test accuracy, balanced accuracy, macro F1
                         (benchkit.child.metrics, the bench's definitions)
  agree_with, agree      the paired route (lsqr_default <-> dual_forced,
                         lsqr_fixed <-> exact_fixed) and the fraction of
                         test predictions the two share
  coef_rel_dist          ||coef_lsqr - coef_exact||_F / ||coef_exact||_F of
                         the pair (float64 of the float32 coef_)
  icpt_rel_dist          the same for intercept_
  fit_seconds            wall time of fit (informative only: untimed)
  n_threads, error       error = repr of the exception ("not run: ..."
                         when a route the unit depends on failed)
  n_channels, m, solver (solver_), cv_score (cv_score_), f1w (weighted F1),
  dec_rel_dist (the relative Frobenius distance of the pair's test decision
  matrices), mem_peak_rss_mb (peak process RSS over fit and predict, MB =
  1e6 bytes), bench_acc (lsqr_default only: the bench's FERRETS accuracy of
  the same (dataset, fold), measured at 10 threads), checks, detail (JSON:
  the alpha grid, alpha_scores_, the per-class step counts and stop reasons
  by phase, the cap and tolerance, the lockstep group sizes, the DUAL_MAX_N
  in effect, predict seconds, fit_stats_), started_at.

Built-in checks (the ``checks`` column, "name=ok|FAIL|...|na" joined by
";"; a FAIL is also printed):
  solver                 lsqr_* routes "lsqr", the dual ones "gcv"
  features               n_features_ and feature_offsets_ equal across the
                         routes (one design)
  iters_multiset         lsqr_* routes: the per-class step counts equal the
                         exact multiset from the forward products
  no_lsqr_solves         the dual routes ran no LSQR solve
  dual_max_n_restored    every route: ferrets.config.DUAL_MAX_N is the
  patches_restored       value it had before the fit, and the lsqr module
                         holds its own functions again
  bench_acc              lsqr_default's accuracy equals the bench row's
                         ("differs(+x)": the bench ran at 10 threads)
  alpha, lambda0         the fixed routes ran at lsqr_default's alpha_ with
                         prior_lambda_ 0
  coef_eq_lsqr_default   lsqr_fixed's coef_ is lsqr_default's final solve
                         (bit for bit; "near(x)" names a rounding-size
                         difference), with iters_eq_lsqr_default_final and
                         icpt_eq_lsqr_default_shifted
  coef_eq_dual_forced    exact_fixed's coef_ equals dual_forced's when
                         dual_forced chose the same alpha ("na" otherwise)

Resumable: the CSV is append-only, a unit's four rows are appended and
fsynced together once the unit is over, and a unit with any row in the CSV is
skipped (delete its rows to re-run it). A route that raises is that route's
error row; a unit whose data do not load is four error rows.

usage:
  lsqr_exact.py [--benchmarks ucr,multiverse] [--datasets D,E|bench:D]
        [--folds 0,1,2] [--threads 6] [--dual-max-n 10000]
        [--out results/lsqr_exact.csv] [--no-warm] [--dry-run]
  --folds applies to every selected benchmark (default: UCR 0,1,2,
  Multiverse 0). The released rows ran at 6 threads.
"""

import os
import sys

THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMEXPR_MAX_THREADS")
DEFAULT_THREADS = 6        # the released rows' thread count (untimed)


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
    _t = _argv_value("--threads", str(DEFAULT_THREADS))
    try:
        _t = str(max(1, int(_t)))
    except ValueError:
        _t = str(DEFAULT_THREADS)       # argparse reports the bad value
    for _var in THREAD_ENV:
        os.environ[_var] = _t

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "lsqr_exact.csv"
BENCH_CSV = {"ucr": RESULTS / "bench_ucr.csv",
             "multiverse": RESULTS / "bench_multiverse.csv"}
REF_MODEL = "FERRETS"             # the shipped batch arm's bench rows
BENCHMARKS = ("ucr", "multiverse")
# the problems with 3000 < n_train <= 10000 (FERRETS fold-0 rows of
# bench_ucr.csv / bench_multiverse.csv), smallest n_train first
UCR_SET = ("FordA", "FordB", "Crop", "ElectricDevices")
MV_SET = ("SitStand", "PhonemeSpectra", "BenzeneConcentration_disc",
          "LiveFuelMoistureContent_disc", "BIDMC32RR_disc",
          "BIDMC32SpO2_disc", "BIDMC32HR_disc", "FaceDetection",
          "Locust2022", "SpokenArabicDigits", "PenDigits", "UCIActivity",
          "CrowdSourced")
DEFAULT_SETS = {"ucr": UCR_SET, "multiverse": MV_SET}
DEFAULT_FOLDS = {"ucr": (0, 1, 2), "multiverse": (0,)}
DUAL_FORCE_N = 10000
ROUTES = ("lsqr_default", "dual_forced", "lsqr_fixed", "exact_fixed")
PAIR = {"lsqr_default": "dual_forced", "dual_forced": "lsqr_default",
        "lsqr_fixed": "exact_fixed", "exact_fixed": "lsqr_fixed"}
DUAL_ROUTES = ("dual_forced", "exact_fixed")
FIXED_ROUTES = ("lsqr_fixed", "exact_fixed")
COLUMNS = [
    "benchmark", "dataset", "fold", "route", "n_train", "n_test",
    "n_classes", "n_features", "alpha", "prior_lambda", "iters_max",
    "iters_mean", "cap_hits", "n_class_solves", "acc", "bacc", "f1m",
    "agree_with", "agree", "coef_rel_dist", "icpt_rel_dist", "fit_seconds",
    "n_threads", "error",
    # extras
    "n_channels", "m", "solver", "cv_score", "f1w", "dec_rel_dist",
    "mem_peak_rss_mb", "bench_acc", "checks", "detail", "started_at"]
KEY = ("benchmark", "dataset", "fold")
NEAR_REL = 1e-6        # a coefficient difference this small is rounding
# --dry-run estimate only (seconds; rough). One-thread rates: syrk 4.1e10
# n^2 F per second (n = 1500-2000, F = 20000), scipy eigh evd 0.15-0.18 s
# per 1e9 n^3 (n = 1500-2500); the eigendecomposition's tridiagonal
# reduction is bound by memory, so it is taken to scale to 3 threads at
# most. The transform, the LSQR solves and predict are the banked FERRETS
# times (10 threads), scaled by 10 / threads.
GRAM_RATE_1T = 3.5e10
EIGH_S_1T = 0.2e-9
EIGH_MAX_SPEEDUP = 3.0
TS_PARSE_BPS = 15e6    # .ts bytes benchkit's prefetch parses per second


# ---------------------------------------------------------------- utils ---

def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _now():
    return datetime.datetime.now()


def _stamp():
    return f"{_now():%Y-%m-%d %H:%M:%S}"


@contextlib.contextmanager
def patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _dumps(obj):
    return json.dumps(obj, default=_json_default, separators=(",", ":"),
                      sort_keys=True)


def rel_dist(a, b):
    """||a - b||_F / ||b||_F in float64 (b: the exact route's); None when
    the shapes differ or b is zero."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    nb = float(np.linalg.norm(b))
    if nb == 0.0:
        return None
    return float(np.linalg.norm(a - b)) / nb


# ------------------------------------------------------- instrumentation ---

class _CountingOp:
    """The operator _lsqr_group sees: the route's own _CenteredOp, every call
    passed through unchanged, with a log of how many columns each forward
    product carries (one product per lockstep pass, over the columns still
    active)."""

    def __init__(self, op, log):
        self._op = op
        self._log = log
        self.shape = op.shape

    def matvec(self, V):
        self._log.append(int(V.shape[1]))
        return self._op.matvec(V)

    def rmatvec(self, U):
        return self._op.rmatvec(U)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._op, name)


class LsqrLog:
    """The LSQR class solves of one fit. While active it replaces, in
    ferrets.classification.lsqr (the module that calls them),
    _lsqr_holdout and _lsqr_solve, and their weighted twins _lsqr_holdout_ops
    and _lsqr_solve_ops (to name the phase), _lsqr_group (one
    record per lockstep group: its G class columns, the cap, the tolerance,
    the dampings, the active columns of every forward product) and _gk_store
    (one call per extra Golub-Kahan step of a column that continues). Every
    wrapper calls the original with the same arguments and returns its
    result."""

    PHASES = (("_lsqr_holdout", "holdout"), ("_lsqr_solve", "final"),
              ("_lsqr_holdout_ops", "holdout"), ("_lsqr_solve_ops", "final"))
    NAMES = tuple(n for n, _ in PHASES) + ("_lsqr_group", "_gk_store")

    def __init__(self):
        import ferrets.classification.lsqr as mod
        self.mod = mod
        self.groups = []
        self._phase = None
        self._cur = None
        self._orig = None

    def __enter__(self):
        import numpy as np
        mod = self.mod
        orig = {k: getattr(mod, k) for k in self.NAMES}
        self._orig = orig
        log = self

        def phased(name, phase):
            def wrapper(*a, **kw):
                prev, log._phase = log._phase, phase
                try:
                    return orig[name](*a, **kw)
                finally:
                    log._phase = prev
            return wrapper

        def group(op, B, damps_sq, tol, max_iter, X):
            G = int(B.shape[1])
            rec = dict(phase=log._phase, G=G, max_iter=int(max_iter),
                       tol=float(tol), n_damps=int(len(damps_sq)), act=[],
                       store=[0] * G,
                       nonzero=[bool(np.any(B[:, g] != 0)) for g in range(G)])
            prev, log._cur = log._cur, rec
            try:
                return orig["_lsqr_group"](_CountingOp(op, rec["act"]), B,
                                           damps_sq, tol, max_iter, X)
            finally:
                log._cur = prev
                log.groups.append(rec)

        def store(vn, al, Brow, Vv, g):
            rec = log._cur
            if rec is not None:
                rec["store"][int(g)] += 1
            return orig["_gk_store"](vn, al, Brow, Vv, g)

        for k, f in ([(n, phased(n, ph)) for n, ph in self.PHASES]
                     + [("_lsqr_group", group), ("_gk_store", store)]):
            setattr(mod, k, f)
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            for k, f in self._orig.items():
                setattr(self.mod, k, f)
        self._orig = None
        return False


def group_counts(rec):
    """(per-column step counts, the exact multiset, whether they agree) of
    one lockstep group record. The multiset: a_t columns took part in the
    t-th forward product, so a_t - a_{t+1} columns took exactly t steps and
    G - a_1 none (a zero target). Per column: a column that was active took
    one step plus one per _gk_store call (a stop on the tests, on the cap or
    on an exact breakdown all leave it so)."""
    G = rec["G"]
    act = list(rec["act"])
    ext = act + [0]
    multi = [0] * max(0, G - (act[0] if act else 0))
    for t in range(len(act)):
        multi += [t + 1] * max(0, ext[t] - ext[t + 1])
    per = [rec["store"][g] + 1 if rec["nonzero"][g] else 0 for g in range(G)]
    return per, multi, sorted(per) == sorted(multi)


def solve_stats(groups):
    """Iteration summary of one fit's LSQR groups: (stats for the row
    columns, per-phase per-class counts for the detail, multiset check).
    The detail's n_unnamed counts the groups no named entry point ran
    ("unknown" phase): unit_checks fails the unit on any."""
    by_phase, stops = {}, {}
    allc, caps, ok = [], 0, True
    caps_phase = {}
    for rec in groups:
        per, multi, agree = group_counts(rec)
        ok = ok and agree
        counts = per if agree else sorted(multi)
        ph = rec["phase"] or "unknown"
        by_phase.setdefault(ph, []).extend(counts)
        # per class (class order when the two counts agree): "cap" when the
        # solve reached RIDGE_MAX_ITER, "tol" below it (the tests passed, or
        # an exact breakdown), "zero" for a zero target column (no step)
        stops.setdefault(ph, []).extend(
            "cap" if x >= rec["max_iter"] else ("zero" if x == 0 else "tol")
            for x in counts)
        c = sum(1 for x in multi if x >= rec["max_iter"])
        caps += c
        caps_phase[ph] = caps_phase.get(ph, 0) + c
        allc.extend(multi)
    out = dict(n_class_solves=len(allc), cap_hits=caps,
               iters_max=max(allc) if allc else None,
               iters_mean=(sum(allc) / len(allc)) if allc else None)
    info = dict(iters=by_phase, stops=stops, cap_hits_by_phase=caps_phase,
                max_iter=sorted({r["max_iter"] for r in groups}),
                tol=sorted({r["tol"] for r in groups}),
                n_damps={(r["phase"] or "unknown"): r["n_damps"]
                         for r in groups},
                n_unnamed=sum(1 for r in groups if r["phase"] is None),
                group_sizes=[r["G"] for r in groups])
    return out, info, ok


# --------------------------------------------------------------- routes ---

def route_overrides(route, alpha1):
    """The parameters a route sets over FERRETS_PARAMS."""
    if route in FIXED_ROUTES:
        return dict(alphas=(float(alpha1),), prior_shift=False)
    return {}


def run_route(route, fold, n_threads, over, dual_n):
    """Fit the FERRETS model with ``over`` on the fold (DUAL_MAX_N set to
    ``dual_n`` during the fit when it is not None), its test decisions and
    predictions (classes_[_decide(decision)], what predict returns), and
    what the rows and checks read. The model itself is dropped."""
    import numpy as np
    import ferrets.config as fcfg
    from ferrets.classification.common import _decide
    from experiments.benchkit.arms import _ferrets
    from experiments.benchkit.child import metrics
    from experiments.benchkit.memory import RssSampler, rss_mb
    gc.collect()
    shipped = fcfg.DUAL_MAX_N
    base = rss_mb()
    sampler = RssSampler(interval_s=0.1).start()
    log = LsqrLog()
    try:
        model = _ferrets(int(n_threads), **over)
        with contextlib.ExitStack() as stack:
            if dual_n is not None:
                stack.enter_context(patched(fcfg, "DUAL_MAX_N", int(dual_n)))
            in_effect = int(fcfg.DUAL_MAX_N)
            stack.enter_context(log)
            t0 = timer()
            model.fit(fold.X_train, fold.y_train)
            fit_s = timer() - t0
        t1 = timer()
        d = model.decision_function(fold.X_test)
        pred_s = timer() - t1
    finally:
        mem = sampler.stop()
    dual_restored = fcfg.DUAL_MAX_N == shipped
    patches_restored = all(
        getattr(getattr(log.mod, k), "__module__", None) == log.mod.__name__
        for k in LsqrLog.NAMES[:-1])
    dec = np.asarray(d, dtype=np.float64).reshape(len(fold.y_test), -1)
    pred = model.classes_[_decide(dec)]
    met = metrics(fold.y_test, pred)
    st, info, iters_ok = solve_stats(log.groups)
    stats = {k: v for k, v in (getattr(model, "fit_stats_", None) or {}).items()
             if v is None or isinstance(v, (bool, int, float, str,
                                            np.integer, np.floating))}
    a_sc = getattr(model, "alpha_scores_", None)
    cv = getattr(model, "cv_score_", None)
    lam = getattr(model, "prior_lambda_", None)
    out = dict(
        route=route, pred=pred, dec=dec,
        coef=np.array(model.coef_, copy=True),
        icpt=np.array(model.intercept_, dtype=np.float64, copy=True),
        offsets=np.array(model.transformer_.feature_offsets_, copy=True),
        n_features=int(model.transformer_.n_features_), alpha=float(model.alpha_),
        lam=None if lam is None else float(lam), solver=str(model.solver_),
        cv=None if cv is None else float(cv), fit_s=fit_s, pred_s=pred_s,
        mem_peak=mem["peak_rss_mb"], mem_median=mem["median_rss_mb"],
        mem_base=base, dual_restored=bool(dual_restored),
        patches_restored=bool(patches_restored), iters_ok=bool(iters_ok),
        phases_named=info["n_unnamed"] == 0, **st, **met)
    out["detail"] = dict(
        alphas=list(model.alphas), prior_shift=bool(model.prior_shift),
        dual_max_n=in_effect,
        alpha_scores=None if a_sc is None else [float(s) for s in a_sc],
        predict_seconds=pred_s, mem_base_rss_mb=base,
        mem_median_rss_mb=mem["median_rss_mb"], fit_stats=stats,
        **(info if log.groups else {}))
    out["iters_final"] = info["iters"].get("final") if log.groups else None
    del model
    gc.collect()
    return out


def _cmp(a, b):
    """'ok' when the arrays are equal bit for bit, 'near(x)' when the
    relative difference is rounding-sized, else 'FAIL(x)'."""
    import numpy as np
    if a.shape != b.shape:
        return "FAIL(shape)"
    if np.array_equal(a, b):
        return "ok"
    r = rel_dist(a, b)
    r = float("inf") if r is None else r
    return f"near({r:.1e})" if r <= NEAR_REL else f"FAIL({r:.1e})"


def unit_checks(res, alpha1, ybar, bench_acc):
    """{route: {check: status}} over the routes that ran (module docstring,
    "Built-in checks")."""
    import numpy as np

    def ok(b):
        return "ok" if b else "FAIL"
    chk = {r: {} for r in ROUTES}
    ref = next((res[r] for r in ROUTES if r in res), None)
    for r, x in res.items():
        chk[r]["solver"] = ok(x["solver"] == ("gcv" if r in DUAL_ROUTES
                                              else "lsqr"))
        chk[r]["features"] = ok(x["n_features"] == ref["n_features"]
                                and np.array_equal(x["offsets"],
                                                   ref["offsets"]))
        if r in DUAL_ROUTES:
            chk[r]["no_lsqr_solves"] = ok(x["n_class_solves"] == 0)
        else:
            chk[r]["iters_multiset"] = ok(x["iters_ok"])
            chk[r]["phases_named"] = ok(x["phases_named"])
        chk[r]["dual_max_n_restored"] = ok(x["dual_restored"])
        chk[r]["patches_restored"] = ok(x["patches_restored"])
    d = res.get("lsqr_default")
    s = res.get("dual_forced")
    f3 = res.get("lsqr_fixed")
    e4 = res.get("exact_fixed")
    if d is not None:
        if bench_acc is None:
            chk["lsqr_default"]["bench_acc"] = "na"
        else:
            dd = d["acc"] - bench_acc
            chk["lsqr_default"]["bench_acc"] = ("ok" if abs(dd) <= 1e-12
                                                else f"differs({dd:+.5f})")
    for r, x in (("lsqr_fixed", f3), ("exact_fixed", e4)):
        if x is None:
            continue
        chk[r]["alpha"] = ok(alpha1 is not None and x["alpha"] == alpha1)
        chk[r]["lambda0"] = ok(x["lam"] == 0.0)
    if f3 is not None and d is not None:
        chk["lsqr_fixed"]["coef_eq_lsqr_default"] = _cmp(f3["coef"],
                                                         d["coef"])
        chk["lsqr_fixed"]["iters_eq_lsqr_default_final"] = ok(
            f3["iters_final"] == d["iters_final"])
        lam = d["lam"] or 0.0
        chk["lsqr_fixed"]["icpt_eq_lsqr_default_shifted"] = ok(
            f3["icpt"].shape == d["icpt"].shape and bool(np.allclose(
                f3["icpt"] - lam * ybar, d["icpt"], rtol=0.0, atol=1e-12)))
    if e4 is not None and s is not None:
        chk["exact_fixed"]["coef_eq_dual_forced"] = (
            _cmp(e4["coef"], s["coef"]) if s["alpha"] == alpha1 else "na")
    return chk


def _ybar(y_train):
    """The training targets' means, as the prior shift reads them
    (ferrets.classification.common._targets of fit's label indices): the
    class-weighted means when the shipped head weights the classes
    (FERRETS_PARAMS' batch_class_weight, on imbalanced classes), the plain
    ones otherwise."""
    import numpy as np
    from experiments.benchkit.arms import FERRETS_PARAMS
    from ferrets.classification.common import _targets
    from ferrets.classification.weights import _ridge_class_weights
    classes, y_idx = np.unique(np.asarray(y_train), return_inverse=True)
    y_idx = y_idx.astype(np.int64).ravel()
    Y = _targets(y_idx, len(classes))
    w = _ridge_class_weights(y_idx, len(classes),
                             FERRETS_PARAMS["batch_class_weight"])
    if w is None:
        return Y.mean(axis=0)
    wr = w[y_idx]
    return (wr[:, None] * Y).sum(axis=0) / wr.sum()


def _blank_row(unit, route, n_threads, started):
    row = {c: None for c in COLUMNS}
    row.update(benchmark=unit["benchmark"], dataset=unit["dataset"],
               fold=int(unit["fold"]), route=route,
               n_train=unit["n_train"], n_test=unit["n_test"],
               n_channels=unit["n_channels"], m=unit["m"],
               agree_with=PAIR[route], n_threads=int(n_threads),
               started_at=started)
    return row


def run_unit(unit, fold, n_threads, dual_n, bench_acc, log=print):
    """The four routes on one loaded fold: four rows."""
    import numpy as np
    started = _now().isoformat(timespec="seconds")
    rows = {r: _blank_row(unit, r, n_threads, started) for r in ROUTES}
    for r in ROUTES:
        rows[r].update(n_train=int(len(fold.y_train)),
                       n_test=int(len(fold.y_test)),
                       n_classes=int(fold.n_classes),
                       n_channels=int(fold.X_train.shape[1]),
                       m=int(fold.X_train.shape[2]))
    res = {}
    alpha1 = None
    for r in ROUTES:
        if r in FIXED_ROUTES and alpha1 is None:
            rows[r]["error"] = "not run: lsqr_default failed (no alpha)"
            continue
        t0 = timer()
        try:
            x = run_route(r, fold, n_threads, route_overrides(r, alpha1),
                          dual_n if r in DUAL_ROUTES else None)
        except Exception as error:  # noqa: BLE001 -- bank it, keep going
            traceback.print_exc()
            sys.stdout.flush()
            rows[r]["error"] = repr(error)[:400]
            log(f"  {r}: ERROR after {timer() - t0:.1f} s: {error!r}"[:300])
            continue
        res[r] = x
        if r == "lsqr_default":
            alpha1 = x["alpha"]
        its = (f" steps max {x['iters_max']} mean {x['iters_mean']:.1f} "
               f"caps {x['cap_hits']}/{x['n_class_solves']}"
               if x["n_class_solves"] else "")
        log(f"  {r}: {x['solver']} alpha {x['alpha']:g} lambda "
            f"{x['lam']:g} acc {x['acc']:.4f} bacc {x['balanced_acc']:.4f} "
            f"f1m {x['f1_macro']:.4f}{its} fit {x['fit_s']:.1f} s pred "
            f"{x['pred_s']:.1f} s rss {x['mem_peak'] / 1000:.2f} GB")
    ybar = _ybar(fold.y_train)
    chk = unit_checks(res, alpha1, ybar, bench_acc)
    for r, x in res.items():
        row = rows[r]
        row.update(n_features=x["n_features"], alpha=x["alpha"],
                   prior_lambda=x["lam"], iters_max=x["iters_max"],
                   iters_mean=x["iters_mean"], cap_hits=x["cap_hits"],
                   n_class_solves=x["n_class_solves"], acc=x["acc"],
                   bacc=x["balanced_acc"], f1m=x["f1_macro"],
                   f1w=x["f1_weighted"], fit_seconds=x["fit_s"],
                   solver=x["solver"], cv_score=x["cv"],
                   mem_peak_rss_mb=x["mem_peak"])
        if r == "lsqr_default":
            row["bench_acc"] = bench_acc
        p = res.get(PAIR[r])
        if p is not None:
            lsq, exa = (x, p) if r not in DUAL_ROUTES else (p, x)
            row.update(
                agree=float(np.mean(np.asarray(x["pred"])
                                    == np.asarray(p["pred"]))),
                coef_rel_dist=rel_dist(lsq["coef"], exa["coef"]),
                icpt_rel_dist=rel_dist(lsq["icpt"], exa["icpt"]),
                dec_rel_dist=rel_dist(lsq["dec"], exa["dec"]))
        row["detail"] = _dumps(x["detail"])
    for r in ROUTES:
        rows[r]["checks"] = ";".join(f"{k}={v}" for k, v in chk[r].items())
    fails = [f"{r}:{k}={v}" for r in ROUTES for k, v in chk[r].items()
             if str(v).startswith("FAIL")]
    return [rows[r] for r in ROUTES], res, fails


# ------------------------------------------------------------- planning ---

def parse_folds(text):
    if text is None:
        return None
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def parse_targets(args):
    """[(benchmark, dataset)] from --benchmarks / --datasets."""
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    bad = [b for b in benches if b not in BENCHMARKS]
    if bad or not benches:
        raise SystemExit(f"--benchmarks: one or more of {BENCHMARKS}, got "
                         f"{args.benchmarks!r}")
    if not args.datasets:
        return [(b, d) for b in benches for d in DEFAULT_SETS[b]]
    out = []
    for item in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        if ":" in item:
            b, d = item.split(":", 1)
            if b not in benches:
                raise SystemExit(f"--datasets {item}: benchmark {b!r} is not "
                                 f"among --benchmarks {benches}")
            out.append((b, d))
            continue
        hits = [b for b in benches if item in DEFAULT_SETS[b]]
        if len(hits) == 1:
            out.append((hits[0], item))
        elif not hits and len(benches) == 1:
            out.append((benches[0], item))
        else:
            raise SystemExit(f"--datasets {item}: not in the default sets of "
                             f"{benches}; name it benchmark:{item}")
    return list(dict.fromkeys(out))


def plan_units(args):
    """Every (benchmark, dataset, fold) unit, smallest n_train first (then
    n_train x C x m, then name and fold). Refuses a problem whose n_train is
    not in (DUAL_MAX_N, --dual-max-n]: below, the shipped route is already
    the exact one; above, the forced dual route is not this driver's."""
    import ferrets.config as fcfg
    from experiments.benchkit import data as bdata
    folds_arg = parse_folds(args.folds)
    units, bad = [], []
    for b, d in parse_targets(args):
        try:
            n_tr, n_te, C, m = bdata.sizes(b, d)
        except Exception as error:  # noqa: BLE001 -- report and stop
            raise SystemExit(f"{b}:{d}: no shape ({error!r})")
        if not (fcfg.DUAL_MAX_N < int(n_tr) <= int(args.dual_max_n)):
            bad.append(f"{b}:{d} (n_train {n_tr})")
            continue
        for f in (folds_arg if folds_arg is not None else DEFAULT_FOLDS[b]):
            units.append(dict(benchmark=b, dataset=d, fold=int(f),
                              n_train=int(n_tr), n_test=int(n_te),
                              n_channels=int(C), m=int(m)))
    if bad:
        raise SystemExit(f"n_train outside ({fcfg.DUAL_MAX_N}, "
                         f"{args.dual_max_n}]: {', '.join(bad)}")
    units.sort(key=lambda u: (u["n_train"], u["n_train"] * u["n_channels"]
                              * u["m"], u["benchmark"], u["dataset"],
                              u["fold"]))
    return units


def bench_reference(log=print):
    """{(benchmark, dataset, fold): FERRETS row} of the bench CSVs (read
    only; bench_acc and the dry-run estimate). A file that cannot be read
    (a bench run rewriting it at that moment) is left out, with a
    warning."""
    import pandas as pd
    out = {}
    for b, path in BENCH_CSV.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as error:  # noqa: BLE001 -- the reference is optional
            log(f"WARNING: {path} unreadable ({error!r}); no bench_acc for "
                f"{b}")
            continue
        df = df[df["model"] == REF_MODEL]
        for r in df.to_dict("records"):
            out[(b, str(r["dataset"]), int(r["fold"]))] = r
    return out


def bench_acc_of(ref, unit):
    r = ref.get((unit["benchmark"], unit["dataset"], int(unit["fold"])))
    if r is None or not _nan(r.get("error")) or _nan(r.get("acc")):
        return None
    return float(r["acc"])


def data_state(unit):
    """Where the unit's data come from before it runs (dry-run only)."""
    from experiments.benchkit import data as bdata
    d = unit["dataset"]
    if unit["benchmark"] == "multiverse":
        if bdata.is_cached(d):
            return "npz", 0
        info = bdata._cached_info(d) or {}
        nb = int(info.get("ts_bytes") or 0)
        if (bdata.data_home() / d).exists():
            return "local .ts -> npz", nb
        return f"download {nb / 1e9:.2f} GB .ts -> npz", nb
    return ("local" if (bdata.data_home() / d).exists() else "download"), 0


def estimate_seconds(unit, ref, threads):
    """Rough compute seconds of one unit (dry-run only; constants above):
    lsqr_default ~ the banked fit, lsqr_fixed ~ 0.7 of it, each dual fit
    half the banked fit (the transform) plus the Gram (n^2 F) and the
    eigendecomposition (n^3), four predicts."""
    r = (ref.get((unit["benchmark"], unit["dataset"], unit["fold"]))
         or ref.get((unit["benchmark"], unit["dataset"], 0)))
    if r is None or not _nan(r.get("error")) or _nan(r.get("n_features")):
        return None
    T = max(1, int(threads))
    s = 10.0 / T
    fit = float(r["time_fit"]) * s
    pred = float(r["time_pred"]) * s
    n = float(unit["n_train"])
    F = float(r["n_features"])
    dual = (0.5 * fit + n * n * F / (GRAM_RATE_1T * T)
            + EIGH_S_1T * n ** 3 / min(T, EIGH_MAX_SPEEDUP))
    return 1.7 * fit + 2.0 * dual + 4.0 * pred


# ------------------------------------------------------------- the bank ---

class Bank:
    """Append-only CSV keyed (benchmark, dataset, fold): a unit's rows are
    appended and fsynced together; a torn last line is closed before the
    next append and skipped when read."""

    def __init__(self, path):
        self.path = Path(path)
        self.keys = set()
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header != COLUMNS:
                    raise SystemExit(f"{self.path}: header differs from this "
                                     "driver's COLUMNS; use another --out")
                for r in reader:
                    if len(r) != len(header):
                        continue
                    rec = dict(zip(header, r))
                    try:
                        self.keys.add((rec["benchmark"], rec["dataset"],
                                       int(rec["fold"])))
                    except ValueError:
                        continue
            with open(self.path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) not in (b"\n", b"\r"):
                    f.write(b"\n")

    def has(self, unit):
        return (unit["benchmark"], unit["dataset"],
                int(unit["fold"])) in self.keys

    def add_rows(self, rows):
        new = not self.path.exists() or not self.path.stat().st_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore",
                               lineterminator="\n")
            if new:
                w.writeheader()
            for row in rows:
                w.writerow({c: ("" if _nan(row.get(c)) else row.get(c))
                            for c in COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        for row in rows:
            self.keys.add((row["benchmark"], row["dataset"], int(row["fold"])))


# ---------------------------------------------------------------- main ----

def warm_up(benchmarks, n_threads, log):
    """JIT and thread pools before the first fit (fit_seconds is
    informative only): the FERRETS arm's own warm-up, both ridge routes,
    on benchkit's warm-up collection (and the 9-channel one when a
    Multiverse unit is planned). It restores ferrets.config before
    returning."""
    import ferrets.config as fcfg
    from experiments.benchkit.arms import _warm_ferrets
    from experiments.benchkit.data import warm_collection
    t0 = timer()
    shipped = fcfg.DUAL_MAX_N
    for b in sorted(set(benchmarks)):
        X, y = warm_collection(b)
        _warm_ferrets(int(n_threads), X, y)
    if fcfg.DUAL_MAX_N != shipped:
        raise RuntimeError("the warm-up left DUAL_MAX_N patched")
    log(f"warm-up done ({timer() - t0:.1f} s)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                   help="comma list of ucr, multiverse (default both)")
    p.add_argument("--datasets", default=None,
                   help="comma list (default: the 4 UCR and 13 Multiverse "
                        "problems); bench:Name for a name outside them")
    p.add_argument("--folds", default=None,
                   help="comma list for every selected benchmark (default "
                        "UCR 0,1,2 and Multiverse 0)")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="numba / BLAS threads and n_jobs (default 6)")
    p.add_argument("--dual-max-n", type=int, default=DUAL_FORCE_N,
                   help="ferrets.config.DUAL_MAX_N of the forced dual fits "
                        "(default 10000)")
    p.add_argument("--out", default=str(OUT),
                   help=f"output CSV (default {OUT})")
    p.add_argument("--no-warm", action="store_true",
                   help="skip the JIT warm-up")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned units and exit")
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("--threads must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)

    def log(msg):
        print(f"{_stamp()} {msg}", flush=True)

    env = {v: os.environ.get(v) for v in THREAD_ENV}
    if any(env[v] != str(args.threads) for v in THREAD_ENV):
        log(f"WARNING: thread environment {env} does not match --threads "
            f"{args.threads} (set before numpy loads only when run as a "
            f"script)")
    units = plan_units(args)
    bank = Bank(args.out)
    todo = [u for u in units if not bank.has(u)]
    ref = bench_reference(log)
    log(f"lsqr_exact: {len(units)} units, {len(units) - len(todo)} "
        f"banked, {len(todo)} to run -> {args.out}; {args.threads} threads, "
        f"forced DUAL_MAX_N {args.dual_max_n}")
    if args.dry_run:
        tot, unknown = 0.0, 0
        for i, u in enumerate(units, 1):
            st, _ = data_state(u)
            e = estimate_seconds(u, ref, args.threads)
            r = (ref.get((u["benchmark"], u["dataset"], u["fold"]))
                 or ref.get((u["benchmark"], u["dataset"], 0)) or {})
            F = r.get("n_features")
            banked = bank.has(u)
            if not banked:
                if e is None:
                    unknown += 1
                else:
                    tot += e
            log(f"  [{i:2d}] {u['benchmark']:10s} {u['dataset']:30s} "
                f"f{u['fold']} n_train {u['n_train']:5d} C {u['n_channels']:3d}"
                f" m {u['m']:5d} F {'?' if _nan(F) else int(F):>7} "
                f"{'BANKED' if banked else 'todo  '} data: {st}; est "
                f"{'?' if e is None else f'{e / 60:.1f} min'}")
        dl = sum(data_state(u)[1] for u in {(u["benchmark"], u["dataset"]): u
                                           for u in todo}.values())
        log(f"estimate: {tot / 3600:.2f} h of compute for {len(todo)} units "
            f"({unknown} without a banked FERRETS row), plus ~"
            f"{dl / TS_PARSE_BPS / 60:.0f} min to parse {dl / 1e9:.1f} GB of "
            f".ts into .npz caches (and its download time); rough")
        print(f"cells to run: {len(todo)}", flush=True)
        return 0
    if not todo:
        log("nothing to run")
        return 0
    import numba
    numba.set_num_threads(int(args.threads))
    if not args.no_warm:
        warm_up([u["benchmark"] for u in todo], args.threads, log)
    from experiments.benchkit import data as bdata
    t_run = timer()
    n_fail_units = 0
    for i, u in enumerate(todo, 1):
        t_u = timer()
        tag = (f"[{i}/{len(todo)}] {u['benchmark']} {u['dataset']} "
               f"f{u['fold']}")
        log(f"{tag}: n_train {u['n_train']}, C {u['n_channels']}, m {u['m']}"
            f" -- loading")
        gen = None
        try:
            gen = bdata.load_folds(u["benchmark"], u["dataset"], [u["fold"]])
            fold = next(gen)
        except Exception as error:  # noqa: BLE001 -- bank it, keep going
            traceback.print_exc()
            sys.stdout.flush()
            started = _now().isoformat(timespec="seconds")
            rows = []
            for r in ROUTES:
                row = _blank_row(u, r, args.threads, started)
                row["error"] = f"load failed: {error!r}"[:400]
                rows.append(row)
            bank.add_rows(rows)
            n_fail_units += 1
            log(f"{tag}: LOAD FAILED ({error!r}); 4 error rows banked; unit "
                f"{timer() - t_u:.1f} s, run {timer() - t_run:.1f} s"[:400])
            continue
        finally:
            if gen is not None:
                gen.close()
        load_s = timer() - t_u
        log(f"{tag}: loaded in {load_s:.1f} s (n_train {len(fold.y_train)}, "
            f"n_test {len(fold.y_test)}, K {fold.n_classes})")
        rows, res, fails = run_unit(u, fold, args.threads, args.dual_max_n,
                                    bench_acc_of(ref, u), log=log)
        d = json.loads(rows[0]["detail"]) if rows[0]["detail"] else None
        if d is not None:
            d["load_seconds"] = load_s
            rows[0]["detail"] = _dumps(d)
        del fold, res           # the routes' coef_ / decisions: banked now
        gc.collect()
        bank.add_rows(rows)
        errs = [r["route"] for r in rows if r["error"]]
        n_fail_units += bool(errs)
        by = {r["route"]: r for r in rows}

        def fmt(v, spec):
            return "-" if _nan(v) else format(v, spec)
        a, b = by["lsqr_default"], by["lsqr_fixed"]
        log(f"{tag}: alpha lsqr {fmt(a['alpha'], 'g')} / dual "
            f"{fmt(by['dual_forced']['alpha'], 'g')}; acc "
            f"{fmt(a['acc'], '.4f')} / {fmt(by['dual_forced']['acc'], '.4f')}"
            f" agree {fmt(a['agree'], '.4f')}; fixed alpha: coef rel "
            f"{fmt(b['coef_rel_dist'], '.2e')} icpt rel "
            f"{fmt(b['icpt_rel_dist'], '.2e')} agree {fmt(b['agree'], '.4f')};"
            f" steps max {fmt(a['iters_max'], 'd')} caps "
            f"{fmt(a['cap_hits'], 'd')}/{fmt(a['n_class_solves'], 'd')}; "
            f"{'errors in ' + ','.join(errs) if errs else 'no error'}; "
            f"{'CHECK FAIL ' + ' '.join(fails) if fails else 'checks ok'}; "
            f"unit {timer() - t_u:.1f} s, run {timer() - t_run:.1f} s")
    log(f"done: {len(todo)} units in {timer() - t_run:.1f} s, "
        f"{n_fail_units} with an error row -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
