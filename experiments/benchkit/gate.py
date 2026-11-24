"""The feasibility gate: predict a cell's fold time and peak memory from the
rows already measured, and do not run the cell when either is out of reach.

WHICH cells are gated is scope.py's business (every slow arm, every MONSTER
and cost cell, and every Multiverse cell of a problem MONSTER also carries,
``scope.always_gated``); this module only predicts and decides. A gated-out
cell is banked as one row per fold with ``error = "not run: <reason>"``, so a
relaunch skips it and the row says it was predicted, not measured. Delete
those rows to force the cell.

The gate learns from the released result files only (``INPUTS``). The paper's
runs were gated on a larger set of rows measured on the same machine, so a
run from scratch can decide a borderline cell differently; every skip the
paper reports is a banked "not run" row of the released files.

The verdict, per (arm, dataset):

1. a failure already measured for the same arm and dataset (any input file
   of the SAME configuration) -> skip, citing it. A failure is keyed by
   (arm, dataset) across the files, so one measured at MONSTER also skips the
   same arm's Multiverse cell of that problem. The transfer is one way: a
   failure measured under the Multiverse protocol on a problem MONSTER also
   carries skips only Multiverse cells, because MONSTER runs its copy under
   its own protocol (its loader holds fewer copies of the data);
2. predicted fold time >= CAP_S -> skip. Per arm, fit and predict time are
   modelled separately as log t = a + b log(N C m) [+ c log N] on the arm's
   completed rows (dataset means; N = train + test); a contracted arm's fit
   is capped at its contract;
3. predicted peak memory >= BUDGET_MB -> skip. Peak = the data the child
   keeps resident while it fits (``resident`` float64 copies) + the arm's
   modelled peak OVER that data (the child's ``mem_peak_mb`` is measured over
   the loaded fold), floored at the copies the loader holds while it builds
   the split. A ``conservative`` decision (the always-gated cells) reads the
   modelled peak over the data at the model's upper error bar (x its
   leave-one-dataset-out factor): a memory under-prediction pages the
   machine, a time under-prediction costs one timeout row;
4. otherwise run. An arm with too few rows to fit a model runs: having
   nothing to predict from is not evidence that a cell is infeasible.

The streaming arms' time model is that of the CACHED streaming fit. A cell
whose cache would not fit (ferrets would then recompute the transform every
epoch) is excluded by the disk rule (benchkit/disk.py) as "not run: disk:
..." rows. bench.py applies this gate first and the disk rule to the cells it
lets through.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .arms import CONTRACT_MINUTES
from .data import multiverse_overlap, sizes, split_kind
from .schema import SKIP_PREFIX, new_row

REPO_ROOT = Path(__file__).resolve().parents[2]
CAP_S = 3600
# of the measuring machine's 68.6 GB (63.9 GiB); the OS, the parent and the
# page cache keep the rest. mem_peak_mb is in MB = 1e6 bytes.
BUDGET_MB = 56_000
MIN_DATASETS = 4
CONTRACT_S = {"cBOSS_c50": 60 * CONTRACT_MINUTES, "TDE_c50": 60 * CONTRACT_MINUTES}
# float64 copies of the whole dataset: (while loading, while fitting).
# MONSTER: monster_loader.load_monster_raw reads the hub file once, front to
# back, in 64 MB chunks straight into the float64 train and test arrays, so
# only the split exists: 1x while loading and while fitting (measured on a
# 640 MB float64 result: 735 MB peak, the result plus a constant ~95 MB).
# UCR/Multiverse: the published split is kept while aeon's stratified
# resample concatenates and re-splits it (3x), and the original plus the
# current fold stay (2x).
COPIES = {"monster": (1, 1)}
DEFAULT_COPIES = (3, 2)

BENCH_CSVS = tuple(f"results/bench_{b}.csv"
                   for b in ("ucr", "multiverse", "monster", "cost"))
# Other released files measured on the same machine under the same 10-thread
# protocol and 1 h cap: (csv, {their arm: our arm}, same configuration?,
# column of peak MB over the data or None). Rows of a DIFFERENT configuration
# (TDE and cBOSS uncontracted) feed only the time model (the contract never
# bound a row that finished within it) and their failures are not this arm's.
LEGACY = (
    ("results/ucr_family_resample10.csv",
     {"MrSQM": "MrSQM", "WEASEL2": "WEASEL2"}, True, None),
    ("results/ucr_family_resample10.csv",
     {"TDE": "TDE_c50", "cBOSS": "cBOSS_c50"}, False, None),
    ("results/uea_muse.csv", {"MUSE_LR": "MUSE_LR"}, True, None),
)
INPUTS = tuple(dict.fromkeys(BENCH_CSVS + tuple(e[0] for e in LEGACY)))
_COLUMNS = ["dataset", "model", "N", "V", "time_fit", "time_pred", "over", "error"]
# An arm that runs the same work as another and differs only in a decision
# rule is predicted from that twin's models, always: the class weights are a
# per-row factor on the SGD update. Rule 1 stays per arm: a twin's failures
# are not this arm's.
COST_TWIN = {"FERRETS_stream_unweighted": "FERRETS_stream"}
# MultiRocket-Hydra runs Hydra's transform and head AND MultiRocket's, so
# Hydra is a LOWER bound for it. PROXY lends Hydra's fitted models while
# MultiRocketHydra has fewer than MIN_DATASETS completed datasets of its own;
# LOWER_BOUND_TWIN: where Hydra's MEASURED cell (same dataset, same N and V)
# is banked, it is a floor on the target's fold time and peak. Hydra's
# measured FAILURES do not transfer.
PROXY = {"MultiRocketHydra": "Hydra"}
LOWER_BOUND_TWIN = {"MultiRocketHydra": "Hydra"}
# Arms whose fit holds a dense design of a width the method fixes: their peak
# over the data is at least the arrays the code keeps alive at once (read in
# aeon 1.5 and scikit-learn 1.8). (d, bytes per value held at once, ridge):
# d columns; the bytes count every copy of the n_train x d design alive
# together; ridge adds RidgeClassifierCV's GCV term, the thin SVD's U (n x d
# float64) when n > d, else the n x n Gram and its eigenvectors.
#  - RDST: 3 features x max_shapelets (10,000); the transform's float64
#    output, the pipeline's scaled float64 copy and the ridge's centred
#    float64 copy: 8 + 8 + 8 bytes.
#  - MultiRocketHydra: MultiRocket's 49,728 float32 columns (Hydra's are left
#    out: the twin floor carries them), scaled in float32, concatenated in
#    float32, cast to float64 by the ridge and centred: 4 + 4 + 8 + 8 bytes.
# It is a lower bound, so it errs toward running.
DENSE_DESIGN = {"RDST": (30_000, 24, True),
                "MultiRocketHydra": (49_728, 24, True)}


def dense_floor_mb(arm, n_train):
    """Peak MB over the data that the arm's dense design alone needs
    (DENSE_DESIGN), or 0 for any other arm."""
    if arm not in DENSE_DESIGN:
        return 0.0
    d, per_value, ridge = DENSE_DESIGN[arm]
    n = float(n_train)
    held = n * d * per_value
    if ridge:
        held += n * d * 8 if n > d else 2 * n * n * 8
    return held / 1e6


@dataclass(frozen=True)
class Decision:
    run: bool
    reason: str
    fold_s: float = None
    peak_mb: float = None


@dataclass(frozen=True)
class _Model:
    coef: np.ndarray
    n: int
    loo: float          # median leave-one-dataset-out error, as a factor


def _normalise(frame, mapping=None, over_col=None):
    f = frame.copy()
    if mapping is not None:
        f = f[f["model"].isin(mapping)].copy()
        f["model"] = f["model"].map(mapping)
    channels = f["n_channels"].fillna(1) if "n_channels" in f else 1
    f["N"] = f["n_train"] + f["n_test"]
    f["V"] = f["N"] * channels * f["m"]
    f["over"] = f[over_col] if over_col and over_col in f else np.nan
    for col in ("time_fit", "time_pred", "error"):
        if col not in f:
            f[col] = np.nan
    return f[_COLUMNS]


# aeon's stratified_resample_data's own internal check: it fires on a
# resampled fold whose two sides do not hold the same classes, a property of
# the resample, not of the arm.
_RESAMPLING_ASSERTION = "assert list(unique_train) == list(unique_test)"
# a fold whose test side carries a class the training side lacks, from a
# label encoder fitted on the training labels only: a loader defect (the
# loader now encodes on the union of the two sides), not a failure of the arm.
_ENCODING_DEFECT = "y contains previously unseen labels"


def _is_measurement(error):
    """A gate verdict, a fold skipped after its sibling failed, or a failure
    of the benchmark's own resampling or label encoding is not a measurement
    of this cell."""
    e = str(error)
    if _RESAMPLING_ASSERTION in e or _ENCODING_DEFECT in e:
        return False
    return not (e.startswith(SKIP_PREFIX) or e.startswith("not attempted"))


def load_measured(root=REPO_ROOT):
    """(completed rows, failures) from ``INPUTS``. ``failures`` maps (arm,
    dataset) to a failure that skips that pair's cell on every benchmark,
    and (arm, dataset, "multiverse") to one measured under the Multiverse
    protocol on a problem MONSTER also carries, which skips its Multiverse
    cell only (module docstring, rule 1)."""
    parts, failures = [], {}
    overlap = frozenset(multiverse_overlap())

    def take(csv, mapping, same, over_col):
        path = Path(root) / csv
        if not path.exists():
            return
        f = _normalise(pd.read_csv(path), mapping, over_col)
        parts.append(f)
        if same:
            multiverse = Path(csv).name.startswith("bench_multiverse")
            for r in f[f["error"].notna()].itertuples():
                if _is_measurement(r.error):
                    key = ((r.model, r.dataset, "multiverse")
                           if multiverse and r.dataset in overlap
                           else (r.model, r.dataset))
                    failures.setdefault(
                        key, f"{str(r.error)[:60]} ({Path(csv).name})")

    for csv in BENCH_CSVS:
        take(csv, None, True, "mem_peak_mb")
    for entry in LEGACY:
        take(*entry)
    # empty frames carry nothing and only make pandas warn about how it will
    # type an all-NA concat
    parts = [p for p in parts if len(p)]
    rows = (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=_COLUMNS))
    return rows[rows["error"].isna()], failures


def _design(N, V, use_n):
    cols = [np.ones(len(V)), np.log(V)]
    if use_n:
        cols.append(np.log(N))
    return np.column_stack(cols)


def _fit(per):
    if len(per) < MIN_DATASETS:
        return None
    N = per["N"].to_numpy(float)
    V = per["V"].to_numpy(float)
    use_n = len(per) >= 10 and np.ptp(np.log(N)) > 1.0
    A, y = _design(N, V, use_n), np.log(per["y"].to_numpy(float))
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    errors = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        c = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        errors.append(abs(A[i] @ c - y[i]))
    return _Model(coef, len(y), float(np.exp(np.median(errors))))


def _eval(model, N, V):
    A = _design(np.array([N], float), np.array([V], float), len(model.coef) == 3)
    return float(np.exp(A[0] @ model.coef))


class Predictor:
    """Fitted lazily per (arm, target) and cached for the process."""

    def __init__(self, rows=None, failures=None, sizes_fn=sizes):
        if rows is None:
            rows, loaded = load_measured()
            failures = loaded if failures is None else failures
        self.rows = rows
        self.failures = failures or {}
        self.sizes = sizes_fn
        self._models = {}
        self._via = {}            # (arm, target) -> the arm whose model it uses

    def _own_model(self, arm, target):
        r = self.rows
        g = r[(r["model"] == arm) & r[target].notna()]
        g = g[(g[target] > 0) & (g["N"] > 0) & (g["V"] > 0)]
        per = (g.groupby("dataset")
               .agg(N=("N", "first"), V=("V", "first"),
                    y=(target, "max" if target == "over" else "mean"))
               .reset_index())
        return _fit(per)

    def model(self, arm, target):
        """Its COST_TWIN's model for a twin arm; else the arm's own model,
        else (too few rows) its PROXY's."""
        key = (arm, target)
        if arm in COST_TWIN:
            self._via[key] = COST_TWIN[arm]
            return self.model(COST_TWIN[arm], target)
        if key not in self._models:
            own = self._own_model(arm, target)
            if own is None and arm in PROXY:
                own = self.model(PROXY[arm], target)
                if own is not None:
                    self._via[key] = PROXY[arm]
            self._models[key] = own
        return self._models[key]

    def borrowed(self, arm):
        """The arms (cost twin or proxy) whose models this arm's predictions
        use."""
        return sorted({v for (a, _), v in self._via.items() if a == arm})

    def predict(self, benchmark, arm, dataset, conservative=False):
        """(fold seconds or None, peak MB) for one cell. ``conservative``
        scales the modelled peak over the data by that model's
        leave-one-dataset-out factor; the data copies stay exact."""
        n_train, n_test, channels, m = self.sizes(benchmark, dataset)
        N = float(n_train + n_test)
        V = N * channels * m
        data_mb = V * 8 / 1e6
        load, resident = COPIES.get(benchmark, DEFAULT_COPIES)
        fit, pred = self.model(arm, "time_fit"), self.model(arm, "time_pred")
        fold_s = None
        if fit is not None and pred is not None:
            fold_s = (min(_eval(fit, N, V), CONTRACT_S.get(arm, math.inf))
                      + _eval(pred, N, V))
        over = self.model(arm, "over")
        over_mb = 0.0
        if over is not None:
            over_mb = _eval(over, N, V) * (over.loo if conservative else 1.0)
        peak = max(load * data_mb, resident * data_mb + over_mb)
        floor = self._twin_floor(arm, dataset, N, V)
        if floor is not None:
            f_s, f_over = floor
            fold_s = f_s if fold_s is None else max(fold_s, f_s)
            peak = max(peak, resident * data_mb + f_over)
        if arm in DENSE_DESIGN:
            peak = max(peak, resident * data_mb + dense_floor_mb(arm, n_train))
        return fold_s, peak

    def _twin_floor(self, arm, dataset, N, V):
        """(fold seconds, peak MB over the data) the arm's LOWER_BOUND_TWIN
        measured on this very cell, or None. The twin's banked rows count
        only at the same N and V, so a dataset name shared by two benchmarks
        cannot lend a measurement of another cell."""
        twin = LOWER_BOUND_TWIN.get(arm)
        if twin is None:
            return None
        r = self.rows
        g = r[(r["model"] == twin) & (r["dataset"] == dataset)
              & (r["N"] == N) & np.isclose(r["V"].astype(float), V)]
        g = g[g["time_fit"].notna() & g["time_pred"].notna()]
        if g.empty:
            return None
        over = g["over"].dropna()
        return (float(g["time_fit"].mean() + g["time_pred"].mean()),
                float(over.max()) if len(over) else 0.0)

    def decide(self, benchmark, arm, dataset, conservative=False):
        """The verdict for one cell (module docstring); ``conservative``
        reads memory at the upper error bar (``predict``)."""
        failed = (self.failures.get((arm, dataset))
                  or self.failures.get((arm, dataset, benchmark)))
        if failed:
            return Decision(False, f"measured before: {failed}")
        fold_s, peak = self.predict(benchmark, arm, dataset, conservative)
        # a prediction from another arm's models says so, in the banked row too
        via = self.borrowed(arm)
        tag = f" [{', '.join(via)}'s models]" if via else ""
        twin = LOWER_BOUND_TWIN.get(arm)
        if twin is not None:
            n_train, n_test, channels, m = self.sizes(benchmark, dataset)
            N = float(n_train + n_test)
            V = N * channels * m
            floor = self._twin_floor(arm, dataset, N, V)
            if floor is not None:
                resident = COPIES.get(benchmark, DEFAULT_COPIES)[1]
                f_peak = resident * V * 8 / 1e6 + floor[1]
                if f_peak >= BUDGET_MB or floor[0] >= CAP_S:
                    what = (f"peak {f_peak / 1e3:,.1f} GB >= the "
                            f"{BUDGET_MB / 1e3:.0f} GB budget"
                            if f_peak >= BUDGET_MB else
                            f"{floor[0]:,.0f} s per fold >= the {CAP_S} s cap")
                    return Decision(
                        False, f"measured on its lower bound {twin}: {what}",
                        fold_s, peak)
        if arm in DENSE_DESIGN:
            n_train, n_test, channels, m = self.sizes(benchmark, dataset)
            resident = COPIES.get(benchmark, DEFAULT_COPIES)[1]
            dense = (resident * (n_train + n_test) * channels * m * 8 / 1e6
                     + dense_floor_mb(arm, n_train))
            if dense >= peak * (1 - 1e-9):
                tag += " [its dense design's floor]"
        over = self.model(arm, "over") if conservative else None
        if over is not None:
            tag += f" (memory at the model's upper error bar x{over.loo:.2f})"
        # memory first: a cell that cannot be held in RAM fails whatever its
        # time, and its reason is the one worth reading
        if peak >= BUDGET_MB:
            return Decision(False, f"predicted peak {peak / 1e3:,.1f} GB >= "
                                   f"the {BUDGET_MB / 1e3:.0f} GB budget{tag}",
                            fold_s, peak)
        if fold_s is not None and fold_s >= CAP_S:
            return Decision(False, f"predicted {fold_s:,.0f} s per fold >= "
                                   f"the {CAP_S} s cap{tag}", fold_s, peak)
        head = ("no time model (too few rows)" if fold_s is None
                else f"predicted {fold_s:,.0f} s per fold")
        return Decision(True, f"{head}, peak {peak / 1e3:,.1f} GB{tag}",
                        fold_s, peak)

    def describe(self, arm):
        parts = []
        for target in ("time_fit", "time_pred", "over"):
            m = self.model(arm, target)
            via = self._via.get((arm, target))
            parts.append(f"{target} " + (f"n={m.n} x{m.loo:.2f}" if m else "none")
                         + (f" (via {via})" if via else ""))
        return f"{arm}: " + ", ".join(parts)


def skip_rows(cell, decision, n_threads):
    """One banked row per fold the gate declined to run."""
    return [new_row(cell.benchmark, cell.dataset, cell.arm, f,
                    split=split_kind(cell.benchmark, cell.dataset, f),
                    n_threads=int(n_threads), error=SKIP_PREFIX + decision.reason)
            for f in cell.folds]
