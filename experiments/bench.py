"""The benchmark driver: every classifier, every dataset, every fold of the
paper's UCR, Multiverse, MONSTER and cost benchmarks, one protocol.

    bench.py queue    --target all [--dry-run]         the whole run; relaunch = resume
    bench.py run      --benchmark B [--arms A,B] [--datasets D,E] [--folds 0,1]
                      [--gate] [--cap 3600] [--threads 10] [--out CSV] [--force]
    bench.py plan     --benchmark B [same filters]     cells to run and banked counts
    bench.py prefetch --benchmark multiverse [--datasets ...] [--arms ...] [--needed-only]
    bench.py reap     [--dry-run]                      drop caches no unbanked cell reads
    bench.py child    --benchmark B --arm A --dataset D --folds f,f --threads T   (internal)

Protocol: 10 threads for every job, set in the child's environment before
numpy, numba and BLAS are imported; 3600 s per fold (fit + predict) and
3600 s of silence between protocol lines, enforced by the parent, so a
timeout is a row. Fold 0 is the archive's published split (MONSTER's hub
fold 0), fold i >= 1 aeon's stratified resample seeded by i (MONSTER: the
hub's fold i). The cost benchmark is a synthetic size grid plus six real
anchors on their published split, 3 repeats of the same data, ladder-stopped
per arm and axis. Memory is sampled every 20 ms inside the child, one child
per (dataset, arm) running all its folds.

Rows are keyed (dataset, model, fold) and a run skips banked keys, so
deleting rows re-runs those cells. ``queue`` runs the steps of
benchkit/scope.py in order; on the released results a dry run plans nothing.
With ``--gate`` (the queue always passes it) the feasibility gate
(benchkit/gate.py) banks "not run: <prediction>" rows for the slow arms', the
MONSTER and the cost cells it predicts over the hour or the memory budget. A
Multiverse cell of a problem MONSTER also carries is gated in every run, its
memory read at the model's upper error bar, and it waits (not banked) while
its data is not cached. Every run applies the disk rule (benchkit/disk.py):
a cell whose data or FERRETS streaming cache would not fit a healthy disk is
banked once as "not run: disk: ..." and never downloaded; one that would fit
a healthy disk but not the current one waits unbanked, and the queue ends
INCOMPLETE. MrSQM runs in a Linux interpreter (FERRETS_MRSQM_PYTHON, through
WSL on Windows); without one its cells stay unbanked.
"""

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "child":
    # BLAS reads its cap at numpy import and numba at its own import: set
    # the budget before either is loaded (the parent also puts it in env).
    _threads = "10"
    if "--threads" in sys.argv:
        _threads = sys.argv[sys.argv.index("--threads") + 1]
    for _var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = _threads

import argparse  # noqa: E402
import datetime  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchkit import data, disk, parent  # noqa: E402
from experiments.benchkit.arms import ARMS, arms_for  # noqa: E402

DEFAULT_CAP = 3600
DEFAULT_THREADS = 10


def bench_csv(benchmark, results_dir=None):
    return parent.bank_path(benchmark, results_dir)


def out_path(args):
    if getattr(args, "out", None):
        return pathlib.Path(args.out)
    return bench_csv(args.benchmark)


def _csv_list(text):
    return [t.strip() for t in text.split(",") if t.strip()] if text else []


def resolve_arms(args):
    """The arms declared on the benchmark, or the ones --arms names."""
    declared = arms_for(args.benchmark)
    wanted = _csv_list(args.arms) or declared
    bad = [a for a in wanted if a not in ARMS]
    off = [a for a in wanted if a in ARMS and a not in declared]
    if bad or off:
        raise SystemExit(f"unknown arms {bad}; not declared on "
                         f"{args.benchmark}: {off}. Declared: {declared}")
    return wanted


def resolve_datasets(args):
    known = data.datasets(args.benchmark)
    wanted = _csv_list(args.datasets)
    if not wanted:
        return known
    bad = [d for d in wanted if d not in known]
    if bad:
        raise SystemExit(f"unknown datasets for {args.benchmark}: {bad}")
    return [d for d in known if d in wanted]   # keep the cost order


def resolve_folds(args, arms):
    """{arm: folds}: --folds for every arm, else the benchmark's default."""
    given = tuple(int(f) for f in _csv_list(args.folds))
    return {a: given or data.default_folds(args.benchmark, a) for a in arms}


def _fmt(row):
    def num(key, fmt):
        v = row.get(key)
        return fmt % v if isinstance(v, (int, float)) and v == v else "   -"
    line = (f"  {row['dataset']:>26s} {row['model']:11s} f{int(row['fold']):<2d} "
            f"acc={num('acc', '%.4f')} f1={num('f1_macro', '%.4f')} "
            f"fit={num('time_fit', '%8.2f')}s pred={num('time_pred', '%7.2f')}s "
            f"mem={num('mem_peak_mb', '%7.0f')}MB (med {num('mem_median_mb', '%.0f')})")
    if row.get("error"):
        line += f"  ERROR: {str(row['error'])[:70]}"
    return line


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cmd_child(args):
    from experiments.benchkit.child import run_cell
    run_cell(args.benchmark, args.arm, args.dataset,
             [int(f) for f in args.folds.split(",")], int(args.threads))


def _cells(args):
    bank = parent.CsvBank(out_path(args))
    arms = resolve_arms(args)
    cells = parent.plan_cells(args.benchmark, arms, resolve_datasets(args),
                              resolve_folds(args, arms), bank.keys())
    # an arm that runs on a named subset of a benchmark (the batch head at
    # MONSTER) never plans a cell outside it, whoever launches the run
    from experiments.benchkit import scope
    cells = [c for c in cells
             if c.dataset in scope.RESTRICTED.get(c.benchmark, {}).get(
                 c.arm, (c.dataset,))]
    if args.benchmark == "cost":
        from experiments.benchkit.cost import filter_cells
        cells = filter_cells(cells)
    return bank, cells


def cmd_plan(args):
    bank, cells = _cells(args)
    n_rows = sum(len(c.folds) for c in cells)
    banked = bank.keys()
    print(f"{args.benchmark}: {len(cells)} cells ({n_rows} fold-rows) to run, "
          f"{len(banked)} rows banked in {bank.path}")
    df = bank.frame()
    done = df.groupby("model").size().to_dict() if len(df) else {}
    arms = resolve_arms(args)
    per_arm = resolve_folds(args, arms)
    for arm in arms:
        n = sum(1 for c in cells if c.arm == arm)
        folds = sum(len(c.folds) for c in cells if c.arm == arm)
        print(f"  {arm:11s} {n:4d} cells / {folds:5d} fold-rows to run, "
              f"{done.get(arm, 0):5d} banked, folds {list(per_arm[arm])}")
    if args.verbose:
        for c in cells:
            print(f"    {c.dataset:>26s} {c.arm:11s} folds {list(c.folds)}")
    if args.gate:
        _plan_gate(cells)


def _plan_gate(cells):
    """What ``run --gate`` would do with each planned cell, read-only: the
    feasibility gate's verdict (predicted seconds per fold, predicted peak),
    then, for a cell it lets through, the disk rule's. Nothing is banked and
    nothing runs."""
    from experiments.benchkit import scope
    from experiments.benchkit.gate import Predictor
    predictor = Predictor()
    print("gate and disk rule, as `run --gate` applies them (nothing banked):")
    for c in cells:
        if scope.gated(c.benchmark, c.arm, c.dataset):
            d = predictor.decide(c.benchmark, c.arm, c.dataset,
                                 conservative=scope.always_gated(c.benchmark,
                                                                 c.dataset))
            fold_s = "-" if d.fold_s is None else f"{d.fold_s:,.0f} s"
            peak = "-" if d.peak_mb is None else f"{d.peak_mb / 1e3:,.1f} GB"
            verdict = "RUN" if d.run else "GATE SKIP"
            line = (f"  {c.dataset:>26s} {c.arm:18s} {verdict:9s} "
                    f"fold {fold_s:>9s} peak {peak:>9s}")
            run = d.run
            if not run:
                line += f"  {d.reason}"
        else:
            line = f"  {c.dataset:>26s} {c.arm:18s} {'RUN':9s} not gated"
            run = True
        if run:
            v = disk.decide(c.benchmark, c.arm, c.dataset)
            if v.wait:
                line += f"  DISK WAIT: {v.reason}"
            elif not v.run:
                line += f"  DISK EXCLUDE: {v.reason}"
            elif v.borderline:
                line += "  disk borderline"
            else:
                line += "  disk ok"
        print(line)


def _refuse_if_busy(args):
    if args.force:
        return
    busy = parent.busy_processes()
    if busy:
        raise SystemExit("timed runs must not overlap; still running:\n  "
                         + "\n  ".join(busy) + "\n(--force to override)")


def _reap_orphan_stream_dirs(args):
    """A killed run leaves its streaming fit's temp-dir cache, and a killed
    loky pool its joblib memmapping folder: sweep the orphans before running
    anything (parent.reap_orphan_stream_dirs keeps a cache a live process
    holds open; reap_orphan_joblib_dirs drops only a dead owner's folder).
    They would otherwise shrink the free disk the disk rule decides on.
    Skipped under --force, which may run beside other drivers."""
    if args.force:
        return
    removed, _ = parent.reap_orphan_stream_dirs()
    if removed:
        print(f"removed {len(removed)} orphaned stream cache(s) from the "
              f"temp dir: {', '.join(removed)}", flush=True)
    removed, _ = parent.reap_orphan_joblib_dirs()
    if removed:
        print(f"removed {len(removed)} orphaned joblib memmapping folder(s) "
              f"from the temp dir: {', '.join(removed)}", flush=True)


DISK_FLOOR_GB = disk.DISK_FLOOR_GB


def _refuse_if_low_disk(args):
    """A cell that fails because the disk is full is not a measurement, and
    its banked key would make every later run skip it: refuse to start below
    the floor instead."""
    if args.force:
        return
    root = (pathlib.Path(args.results_dir) if getattr(args, "results_dir", None)
            else REPO_ROOT / "results")
    while not root.exists() and root.parent != root:
        root = root.parent
    free = shutil.disk_usage(str(root)).free / 2**30
    if free < DISK_FLOOR_GB:
        raise SystemExit(
            f"only {free:.1f} GB free under {root} -- a run on a full disk banks "
            f"failures that are not measurements; free at least "
            f"{DISK_FLOOR_GB:.0f} GB first (--force to override)")


def _disk_rule(cells, bank, n_threads, waiting=None):
    """Bank "not run: disk" rows for the cells whose data or streaming cache
    would exhaust the disk and return the rest (benchkit/disk.py). Decided
    in the parent, so the child of an excluded cell, and the download it
    would have started, never runs. Called just before each cell: the free
    disk grows as the step reaps the caches it is done with, and an
    exclusion is banked for good. A cell that would fit a healthy disk but
    not this one WAITS: nothing banked, nothing run, the cell appended to
    ``waiting`` (the queue ends INCOMPLETE until space is freed)."""
    kept = []
    for cell in cells:
        verdict = disk.decide(cell.benchmark, cell.arm, cell.dataset)
        if verdict.run:
            kept.append(cell)
            continue
        if verdict.wait:
            if waiting is not None:
                waiting.append(cell)
            print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}: "
                  f"WAITING for disk space, not banked -- {verdict.reason}",
                  flush=True)
            continue
        bank.add(disk.skip_rows(cell, verdict, n_threads))
        print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}: "
              f"EXCLUDED, {disk.error_text(verdict.reason)}", flush=True)
    return kept


def _stop_if_below_floor(cell, args):
    """The start-up floor (_refuse_if_low_disk), re-read before every cell
    on every volume the cell touches: the bank's, the temp dir and its
    download root. Below it the step stops here with nothing banked for
    this cell: the queue ends INCOMPLETE and resumes once space is freed.
    Skipped under --force, like the start-up check."""
    if args.force:
        return
    bank_root = pathlib.Path(out_path(args)).resolve().parent
    for path in [bank_root] + disk.cell_volumes(cell.benchmark, cell.dataset):
        free = disk.free_bytes(path)
        if free < disk.FLOOR_BYTES:
            raise SystemExit(
                f"only {free / 2**30:.1f} GB free under {path} before "
                f"{cell.dataset} / {cell.arm} -- below the {DISK_FLOOR_GB:.0f} "
                f"GB floor the step stops, nothing banked for this cell; free "
                f"space and relaunch")


def _no_interpreter(cell, missing):
    """True (and the cell appended to ``missing``) when the cell's arm needs
    an interpreter this machine does not configure: nothing banked."""
    reason = parent.interpreter_missing(cell.arm)
    if reason is None:
        return False
    missing.append(cell)
    print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}: "
          f"{reason}; not banked", flush=True)
    return True


def _report_disk_wait(cell, result, waiting):
    """A child's DISKWAIT: print it and count the cell as waiting."""
    if result.disk_wait is None:
        return
    waiting.append(cell)
    print(f"-- {cell.dataset} / {cell.arm}: WAITING for disk space, the "
          f"remaining folds not banked -- {result.disk_wait}", flush=True)


def _print_waiting(waiting, uncached=(), missing=()):
    if waiting:
        print(f"{len(waiting)} cell(s) WAITING for disk space, left unbanked "
              f"(the disk is below the {disk.gb(disk.HEALTHY_BYTES)} GB healthy "
              f"reference; free space and relaunch): "
              + ", ".join(f"{c.dataset}/{c.arm}" for c in waiting), flush=True)
    if uncached:
        print(f"{len(uncached)} cell(s) WAITING for their data, left unbanked "
              f"(not cached; free space or prefetch, and relaunch): "
              + ", ".join(f"{c.dataset}/{c.arm}" for c in uncached), flush=True)
    if missing:
        print(f"{len(missing)} cell(s) left unbanked, their interpreter is not "
              f"configured ({parent.MRSQM_PYTHON_ENV}): "
              + ", ".join(f"{c.dataset}/{c.arm}" for c in missing), flush=True)


def _waits_for_cache(cell):
    """True for an always-gated cell (scope.always_gated: a Multiverse cell
    of a problem MONSTER also carries) whose cache (.npz and its shape
    .json, data.is_cached) is not complete: it waits for ``bench.py
    prefetch`` rather than download MONSTER-scale data inside the child,
    i.e. inside the fold's time cap."""
    from experiments.benchkit import scope
    return (scope.always_gated(cell.benchmark, cell.dataset)
            and not data.is_cached(cell.dataset))


def _gate(cells, bank, n_threads, only=None):
    """Bank "not run" rows for the cells the gate predicts out of reach and
    return the rest, in order. ``only``: a predicate on a cell; the cells it
    rejects pass through undecided (None: every cell is decided). An
    always-gated cell (scope.always_gated) is decided at the upper memory
    bar."""
    from experiments.benchkit import scope
    from experiments.benchkit.gate import Predictor, skip_rows
    predictor, kept = None, []
    for cell in cells:
        if only is not None and not only(cell):
            kept.append(cell)
            continue
        if predictor is None:
            predictor = Predictor()
        decision = predictor.decide(
            cell.benchmark, cell.arm, cell.dataset,
            conservative=scope.always_gated(cell.benchmark, cell.dataset))
        if decision.run:
            kept.append(cell)
            continue
        bank.add(skip_rows(cell, decision, n_threads))
        print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}: "
              f"not run, {decision.reason}", flush=True)
    return kept


def _results_dir(args):
    """Where the result CSVs live for this invocation: the queue passes the
    bank as --out, a standalone run takes the default."""
    if getattr(args, "results_dir", None):
        return args.results_dir
    if getattr(args, "out", None):
        return str(pathlib.Path(args.out).resolve().parent)
    return None


def _reap_after(cell, args):
    """Drop a dataset's downloaded cache once no unbanked cell reads it.

    Asked per dataset, not per step: the same Multiverse dataset feeds
    several steps and a cost anchor, so the question is whether the whole
    plan is done with it (scope.still_needed). A cell the feasibility gate
    refuses keeps nothing (scope.runnable_by_gate). Only datasets that
    actually have a cache pay for the question."""
    if getattr(args, "keep_cache", False):
        return
    from experiments.benchkit import reap as reaper
    if not reaper.cache_paths(cell.benchmark, cell.dataset):
        return
    from experiments.benchkit import scope
    freed = reaper.reap(cell.benchmark, cell.dataset,
                        scope.still_needed(_results_dir(args),
                                           runnable=scope.runnable_by_gate()))
    if freed:
        print(f"   cache dropped: {cell.dataset} "
              f"({reaper.gb(freed):.2f} GB, re-downloads if needed)", flush=True)


def cmd_run(args):
    _refuse_if_busy(args)
    _reap_orphan_stream_dirs(args)
    _refuse_if_low_disk(args)
    bank, cells = _cells(args)
    if args.benchmark == "cost":
        _run_cost(args, bank, cells)
        return
    # --gate decides the gated cells (slow arms, MONSTER); without it the
    # gate still decides the always-gated ones (a Multiverse cell of a
    # problem MONSTER also carries), whoever launched the run
    from experiments.benchkit import scope
    planned = cells
    cells = _gate(cells, bank, args.threads,
                  only=lambda c: ((args.gate and scope.gated(c.benchmark, c.arm,
                                                             c.dataset))
                                  or scope.always_gated(c.benchmark, c.dataset)))
    # a dataset whose every cell the gate refused is reaped now (its
    # refusals are banked): the per-cell loop below reaps only the datasets
    # it reaches
    kept = {c.dataset for c in cells}
    for name, refused in {c.dataset: c for c in planned}.items():
        if name not in kept:
            _reap_after(refused, args)
    _warn_whatif()
    # the disk rule is decided, and banked, just before each cell, against
    # the disk of that moment: the free disk grows as the step reaps the
    # caches it is done with. Here only a preview.
    preview = [(c, disk.decide(c.benchmark, c.arm, c.dataset)) for c in cells]
    for what, picked in (("excluded", [c for c, v in preview if v.excluded]),
                         ("left waiting (disk below the healthy reference)",
                          [c for c, v in preview if v.wait])):
        if picked:
            print(f"disk rule preview: {len(picked)} of {len(cells)} cells would "
                  f"be {what} on the disk as it is now ("
                  + ", ".join(f"{c.dataset}/{c.arm}" for c in picked[:8])
                  + (", ..." if len(picked) > 8 else "")
                  + "); each is decided just before its turn", flush=True)
    n_rows = sum(len(c.folds) for c in cells)
    print(f"{args.benchmark}: {len(cells)} cells ({n_rows} fold-rows) to run, "
          f"{len(bank.keys())} rows banked, {args.threads} threads, "
          f"cap {args.cap} s per fold -> {bank.path}", flush=True)
    env = parent.child_env(args.threads)
    waiting, uncached, missing = [], [], []
    for i, cell in enumerate(cells):
        run = not _no_interpreter(cell, missing)
        if run:
            _stop_if_below_floor(cell, args)
            run = bool(_disk_rule([cell], bank, args.threads, waiting))
        if run and _waits_for_cache(cell):
            run = False
            uncached.append(cell)
            print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}: "
                  f"WAITING, its data is not cached (bench.py prefetch "
                  f"--benchmark multiverse --arms {cell.arm} --datasets "
                  f"{cell.dataset}); not banked",
                  flush=True)
        if run:
            print(f"-- {cell.dataset} / {cell.arm} folds {list(cell.folds)}",
                  flush=True)

            def on_row(row):
                bank.add([row])
                print(_fmt(row), flush=True)

            result = parent.run_child(parent.child_command(cell, args.threads),
                                      env, args.cap, on_row=on_row,
                                      sweep=not args.force)
            extra = parent.outcome_rows(cell, result, args.threads, args.cap)
            bank.add(extra)
            for row in extra:
                print(_fmt(row), flush=True)
            _report_disk_wait(cell, result, waiting)
        # cells are dataset-major: this was the last arm on this dataset
        if i + 1 == len(cells) or cells[i + 1].dataset != cell.dataset:
            _reap_after(cell, args)
    _print_waiting(waiting, uncached, missing)
    print(f"written: {bank.path}")


def _run_cost(args, bank, cells):
    """Arm-major ladders; a rung that times out, runs out of memory, crashes
    or is gated out stops its axes above it (benchkit/cost.py). A rung the
    disk rule excludes stops nothing: every rung is asked on its own (the
    rule is monotone in size, so the rungs above exclude themselves, each
    with its own numbers). A rung whose interpreter is missing stops nothing
    either: it is left unbanked.

    ``dropped`` is reseeded from the bank at the top: if the process died
    between a rung's failure and the banking of its higher rungs, a relaunch
    re-derives the prune from the banked failure row instead of running
    those rungs."""
    from experiments.benchkit import cost
    from experiments.benchkit.gate import Predictor, skip_rows
    from experiments.benchkit.schema import DISK_PREFIX, SKIP_PREFIX
    predictor = Predictor() if args.gate else None
    queue = cost.order(cells)
    dropped = {}
    banked = bank.frame()
    if len(banked):
        bad = banked[banked["error"].notna()]
        for r in bad.itertuples():
            e = str(r.error)
            # a gate refusal ("not run: predicted ...") also stops its axes;
            # the "not run: rung ..." rows are the pruned ones themselves
            # and must not be treated as a new failure to prune from, and a
            # disk exclusion ("not run: disk ...") stops nothing.
            if r.benchmark == "cost" and (e.startswith(cost.FAIL_PREFIXES) or
                                          (e.startswith(SKIP_PREFIX) and "rung" not in e
                                           and not e.startswith(DISK_PREFIX))):
                failed_cell = parent.Cell("cost", r.dataset, r.model, ())
                for c in cost.pruned_by(queue, failed_cell):
                    dropped.setdefault(c, (failed_cell, "failed (banked)"))
    print(f"cost: {len(queue)} cells, {len(bank.keys())} rows banked, "
          f"{args.threads} threads, cap {args.cap} s per repeat -> {bank.path}",
          flush=True)
    env = parent.child_env(args.threads)
    waiting, missing = [], []
    while queue:
        cell = queue.pop(0)
        if cell in dropped:
            failed_cell, reason = dropped[cell]
            bank.add(cost.skip_rows(cell, failed_cell, reason, args.threads))
            print(f"-- {cell.dataset} / {cell.arm}: not run, rung "
                  f"{failed_cell.dataset} {reason}", flush=True)
            continue
        if _no_interpreter(cell, missing):
            continue
        stopped = None
        excluded = False
        if predictor is not None:
            decision = predictor.decide(cell.benchmark, cell.arm, cell.dataset)
            if not decision.run:
                bank.add(skip_rows(cell, decision, args.threads))
                print(f"-- {cell.dataset} / {cell.arm}: not run, {decision.reason}",
                      flush=True)
                stopped = "gated out"
        if stopped is None:
            # neither an exclusion nor a wait stops the ladder above it
            _stop_if_below_floor(cell, args)
            excluded = not _disk_rule([cell], bank, args.threads, waiting)
        if stopped is None and not excluded:
            print(f"-- {cell.dataset} / {cell.arm} repeats {list(cell.folds)}", flush=True)
            rows = []

            def on_row(row, rows=rows):
                rows.append(row)
                bank.add([row])
                print(_fmt(row), flush=True)
            result = parent.run_child(parent.child_command(cell, args.threads), env,
                                      args.cap, on_row=on_row,
                                      sweep=not args.force)
            extra = parent.outcome_rows(cell, result, args.threads, args.cap)
            bank.add(extra)
            for row in extra:
                print(_fmt(row), flush=True)
            _report_disk_wait(cell, result, waiting)
            # a rung the child banked "not run: disk" (FERRETS's own probe)
            # or left waiting is no cost.failed(): it stops nothing, like
            # the parent's
            if cost.failed(rows + extra, result.timed_out):
                stopped = "timed out" if result.timed_out else "failed"
        if stopped:
            for c in cost.pruned_by(queue, cell):
                dropped.setdefault(c, (cell, stopped))
        # the ladders are arm-major, so an anchor is done only once no cell
        # left in the queue reads it
        if not any(c.dataset == cell.dataset for c in queue):
            _reap_after(cell, args)
    _print_waiting(waiting, missing=missing)
    print(f"written: {bank.path}")


def _reap_unneeded(results_dir, dry_run=False):
    """Drop every downloaded cache (Multiverse .npz, MONSTER hub repo) that
    no unbanked cell reads; a cell the feasibility gate refuses reads
    nothing (scope.runnable_by_gate, as in _reap_after): (keep set, bytes,
    [(benchmark, dataset, bytes)])."""
    from experiments.benchkit import reap as reaper
    from experiments.benchkit import scope
    keep = scope.still_needed(results_dir, runnable=scope.runnable_by_gate())
    freed, dropped = 0, []
    for benchmark in ("multiverse", "monster"):
        for dataset in data.datasets(benchmark):
            n = reaper.reap(benchmark, dataset, keep, dry_run=dry_run)
            if n:
                freed += n
                dropped.append((benchmark, dataset, n))
    return keep, freed, dropped


def _reap_unneeded_caches(args, rdir):
    """At the queue's start, drop the downloaded caches no step reads any
    more: a cache whose last cell was banked just before a kill is read by
    no step, so no _reap_after ever drops it. Skipped under --force, which
    may run beside other drivers reading the same hub cache."""
    if args.force:
        return
    from experiments.benchkit import reap as reaper
    _, freed, dropped = _reap_unneeded(rdir)
    if dropped:
        print(f"dropped {len(dropped)} downloaded cache(s) no step reads, "
              f"{reaper.gb(freed):.1f} GB: "
              + ", ".join(f"{b}:{d}" for b, d, _ in dropped), flush=True)


def cmd_reap(args):
    """Drop every downloaded cache the remaining plan does not read.

    The run does this as it goes and the queue at its start; this sweep is
    for a standalone run that was killed before it could."""
    from experiments.benchkit import reap as reaper
    keep, freed, dropped = _reap_unneeded(args.results_dir, dry_run=args.dry_run)
    verb = "would drop" if args.dry_run else "dropped"
    for benchmark, dataset, n in dropped:
        print(f"  {verb} {benchmark}:{dataset}  {reaper.gb(n):.2f} GB")
    state = "reclaimable" if args.dry_run else "freed"
    print(f"reap: {reaper.gb(freed):.1f} GB {state}; {len(keep)} (benchmark, "
          f"dataset) pairs still read by an unbanked cell")


def _prefetch_cells(args, names):
    """{("multiverse", dataset): [the unbanked cells that read it]}: the
    fold-0 cells of ``--arms`` when given, else every unbanked cell of the
    plan (scope.needed_cells)."""
    from experiments.benchkit import scope
    if not _csv_list(getattr(args, "arms", "")):
        return scope.needed_cells(_results_dir(args))
    arms = resolve_arms(args)
    bank = parent.CsvBank(out_path(args))
    cells = parent.plan_cells("multiverse", arms, names, {a: (0,) for a in arms},
                              bank.keys())
    need = {}
    for cell in cells:
        need.setdefault(("multiverse", cell.dataset), []).append(cell)
    return need


def cmd_prefetch(args):
    """Download and cache the Multiverse datasets the planned cells read.

    A dataset is fetched when at least one of its unbanked cells passes the
    feasibility gate (asked for the gated cells only, at the upper memory
    bar for the always-gated ones, scope.always_gated) and the disk rule.
    The gate's load floor (3 float64 copies) bounds this process's own peak
    (aeon's loader holds about 2.1, data.prefetch). A requested dataset no
    planned cell reads is fetched under the disk rule alone, except a
    problem MONSTER also carries, which is never fetched without a planned
    cell the gate lets through; with ``--arms`` only those arms' fold-0
    cells are planned, and a dataset none of them still needs is skipped,
    as it is under ``--needed-only`` (what the queue's prefetch steps pass
    with their dataset lists) or without ``--datasets``."""
    if args.benchmark != "multiverse":
        raise SystemExit("prefetch caches the Multiverse datasets only (MONSTER "
                         "downloads inside its child; UCR is local)")
    from experiments.benchkit import scope
    from experiments.benchkit.memory import peak_wset_mb
    _warn_whatif()
    names = resolve_datasets(args)
    requested = bool(_csv_list(args.datasets))
    named_arms = bool(_csv_list(getattr(args, "arms", "")))
    # do not re-download what the run has finished with: the caches are
    # dropped as the run goes, so "not cached" does not mean "to fetch"
    need = _prefetch_cells(args, names)
    if not requested or named_arms or getattr(args, "needed_only", False):
        done = [n for n in names if ("multiverse", n) not in need]
        names = [n for n in names if ("multiverse", n) in need]
        if done:
            print(f"prefetch: {len(done)} datasets have no unbanked cell left, "
                  f"not fetched", flush=True)
    cached = sum(data.is_cached(n) for n in names)
    print(f"prefetch: {len(names)} datasets, {cached} already cached", flush=True)
    failed = []
    gate = {}

    def passes_gate(cell):
        """(passes, reason): the gate's verdict on one planned cell."""
        if not scope.gated(cell.benchmark, cell.arm, cell.dataset):
            return True, None
        if "predictor" not in gate:
            from experiments.benchkit.gate import Predictor
            gate["predictor"] = Predictor()
        decision = gate["predictor"].decide(
            cell.benchmark, cell.arm, cell.dataset,
            conservative=scope.always_gated(cell.benchmark, cell.dataset))
        return decision.run, decision.reason

    for i, name in enumerate(names, 1):
        # both files, as the loader requires: an .npz without its .json
        # would make the child call prefetch inside its fold cap
        if data.is_cached(name):
            continue
        cells = need.get(("multiverse", name), [])
        if not cells and scope.always_gated("multiverse", name):
            print(f"  [{i}/{len(names)}] {name}: not downloaded -- no planned cell "
                  f"reads it (a problem MONSTER also carries is fetched only for "
                  f"a cell the gate lets run)", flush=True)
            continue
        # the gate first, as the run applies it: a dataset whose every
        # planned cell is gated out is not downloaded (its cells are banked
        # "not run: <prediction>" when the run step reaches them)
        live, refused = [], []
        for cell in cells:
            ok, reason = passes_gate(cell)
            if ok:
                live.append(cell)
            else:
                refused.append(reason)
        if cells and not live:
            print(f"  [{i}/{len(names)}] {name}: not downloaded -- every planned "
                  f"cell is gated out ({refused[0]})", flush=True)
            continue
        # the disk rule, against the disk as the downloads so far left it:
        # a dataset no planned cell may run is not downloaded (its cells
        # are banked "not run: disk" when the run step reaches them)
        verdicts = ([disk.decide(c.benchmark, c.arm, c.dataset) for c in live]
                    or [disk.decide("multiverse", None, name)])
        if not any(v.run for v in verdicts):
            waits = [v for v in verdicts if v.wait]
            if waits:
                # a healthy disk would hold it: no download now, and the
                # run step leaves its cells unbanked (WAITING) as well
                print(f"  [{i}/{len(names)}] {name}: WAITING for disk space, "
                      f"not downloaded -- {waits[0].reason}", flush=True)
            else:
                print(f"  [{i}/{len(names)}] {name}: EXCLUDED, not downloaded -- "
                      f"{disk.error_text(verdicts[0].reason)}", flush=True)
            continue
        t0 = time.time()
        try:
            info = data.prefetch(name)
        except Exception as error:  # noqa: BLE001 -- its cells will record it
            failed.append(name)
            print(f"  [{i}/{len(names)}] {name}: FAILED {type(error).__name__}: "
                  f"{str(error)[:200]}", flush=True)
            continue
        print(f"  [{i}/{len(names)}] {name}: {info['n_train']}+{info['n_test']} x "
              f"{info['n_channels']} x {info['m']}"
              + ("  HAS NaN" if info.get("has_nan") else "")
              + f"  ({time.time() - t0:.0f} s, process peak "
                f"{peak_wset_mb():,.0f} MB)", flush=True)
    print(f"prefetch: done, {len(failed)} failed"
          + (f": {', '.join(failed)}" if failed else ""), flush=True)


def _warn_whatif():
    """The FREE_ENV what-if is for previews; a run reads the real disk."""
    if disk.whatif_free() is not None:
        print(f"note: {disk.FREE_ENV} is set and IGNORED here -- it is a "
              f"what-if for --dry-run; this run decides on the real disk",
              flush=True)


def _dry_run_verdicts(step, cells, projection=None, get_predictor=None):
    """What a step would do with its unbanked cells: cells per arm, then the
    cells the gate and the disk rule would bank as "not run" instead of
    running (the gate first, as ``bench.py run`` applies them) and the disk
    rule's borderline cells (run; FERRETS's own probe decides). The disk
    verdicts read ``projection`` (disk.project: the free disk when the step
    starts, once the earlier steps' caches are reaped); None reads the disk
    as it is now. An always-gated cell whose data is not cached yet is not
    listed as waiting: the overlap prefetch step precedes its run step.
    ``get_predictor``: a zero-argument callable returning the preview's
    shared Predictor; None builds one here, on the first gated cell."""
    from experiments.benchkit import scope
    predictor = None
    per_arm = {}
    for c in cells:
        per_arm[c.arm] = per_arm.get(c.arm, 0) + 1
    print("    cells per arm: " + ", ".join(f"{a} {n}" for a, n in per_arm.items()),
          flush=True)
    if projection is not None:
        print(f"    free disk at step start, projected: {disk.gb(projection.free)} GB "
              f"(now {disk.gb(projection.now)} + {disk.gb(projection.reclaimed)} "
              f"reaped by then - {disk.gb(projection.downloaded)} fetched by "
              f"earlier steps)", flush=True)
    gated, excluded, borderline, waiting, missing = [], [], [], [], []
    for c in cells:
        reason = parent.interpreter_missing(c.arm)
        if reason is not None:
            missing.append((c, reason))
            continue
        if scope.gated(c.benchmark, c.arm, c.dataset):
            if predictor is None:
                if get_predictor is not None:
                    predictor = get_predictor()
                else:
                    from experiments.benchkit.gate import Predictor
                    predictor = Predictor()
            decision = predictor.decide(
                c.benchmark, c.arm, c.dataset,
                conservative=scope.always_gated(c.benchmark, c.dataset))
            if not decision.run:
                gated.append((c, decision.reason))
                continue
        verdict = (disk.decide_projected(c, projection) if projection is not None
                   else disk.decide(c.benchmark, c.arm, c.dataset))
        if verdict.wait:
            waiting.append((c, verdict.reason))
        elif not verdict.run:
            excluded.append((c, verdict.reason))
        elif verdict.borderline:
            borderline.append((c, verdict.reason))
    for title, items in (("left unbanked, no interpreter", missing),
                         ("gate would not run", gated),
                         ("disk rule would EXCLUDE", excluded),
                         ("disk rule would leave WAITING (below the "
                          f"{disk.gb(disk.HEALTHY_BYTES)} GB healthy reference; "
                          "not banked)", waiting),
                         ("disk rule BORDERLINE (runs; FERRETS's probe "
                          "decides)", borderline)):
        if items:
            print(f"    {title} {len(items)}:", flush=True)
            for c, reason in items:
                print(f"      {c.dataset} / {c.arm}: {reason}", flush=True)


def _queue_dry_run(steps, rdir):
    """Print what each step has left, then the count of unbanked fold-rows
    on the last line, ``cells to run: <N>``. Reads the results, writes
    nothing."""
    from experiments.benchkit import scope
    cells_of = {i: scope.step_cells(s, rdir)
                for i, s in enumerate(steps) if s.kind == "run"}
    total = sum(len(c.folds) for cells in cells_of.values() for c in cells)
    if total:
        plan = [(s, cells_of.get(i)) for i, s in enumerate(steps)]
        now = disk.whatif_free()
        if now is None:
            import tempfile
            now = disk.free_bytes(tempfile.gettempdir())
        # one gate for the whole preview: the projection's predicate and
        # every step's verdicts share its fitted models
        runnable = scope.runnable_by_gate()
    for i, step in enumerate(steps):
        head = f"== [{i + 1}/{len(steps)}] {step.label}"
        if step.kind != "run":
            print(head, flush=True)
            continue
        cells = cells_of[i]
        if not cells:
            print(f"{head}: nothing to run", flush=True)
            continue
        print(f"{head}: {len(cells)} cells / {sum(len(c.folds) for c in cells)} "
              f"fold-rows unbanked", flush=True)
        _dry_run_verdicts(step, cells,
                          disk.project(plan, i, now, runnable=runnable),
                          get_predictor=runnable.predictor)
    print(f"cells to run: {total}", flush=True)


def cmd_queue(args):
    """Run what is unbanked, step by step, from benchkit/scope.py. Each run
    step is its own ``bench.py run`` process, so one step's crash does not
    end the queue, and each step's cells are planned when it starts."""
    from experiments.benchkit import scope
    rdir = args.results_dir
    steps = scope.steps(args.target)
    if args.dry_run:
        print(f"===== queue {args.target}: {len(steps)} steps, dry run {_now()}",
              flush=True)
        _queue_dry_run(steps, rdir)
        return
    _refuse_if_busy(args)
    _refuse_if_low_disk(args)
    me = str(pathlib.Path(__file__).resolve())
    print(f"===== queue {args.target}: {len(steps)} steps, built {_now()}", flush=True)
    _warn_whatif()
    _reap_unneeded_caches(args, rdir)

    def call(argv):
        sys.stdout.flush()
        return subprocess.call([sys.executable, me] + argv, cwd=str(REPO_ROOT))

    for i, step in enumerate(steps, 1):
        head = f"== [{i}/{len(steps)}] {step.label}"
        if step.kind == "prefetch":
            print(f"{head}  {_now()}", flush=True)
            rc = call(["prefetch", "--benchmark", step.benchmark,
                       "--datasets", ",".join(step.datasets), "--needed-only",
                       "--out", str(bench_csv(step.benchmark, rdir))])
            print(f"{head}: exit {rc}  {_now()}", flush=True)
            continue
        cells = scope.step_cells(step, rdir)
        if not cells:
            print(f"{head}: nothing to run", flush=True)
            continue
        print(f"{head}: {len(cells)} cells / {sum(len(c.folds) for c in cells)} "
              f"fold-rows unbanked  {_now()}", flush=True)
        argv = ["run", "--benchmark", step.benchmark, "--arms", ",".join(step.arms),
                "--folds", ",".join(str(f) for f in step.folds),
                "--threads", str(args.threads), "--cap", str(args.cap), "--gate",
                "--out", str(scope.step_path(step, rdir))]
        if step.datasets is not None:
            argv += ["--datasets", ",".join(step.names())]
        if args.force:
            argv.append("--force")
        rc = call(argv)
        print(f"{head}: exit {rc}  {_now()}", flush=True)
    left = [(s.label, sum(len(c.folds) for c in scope.step_cells(s, rdir)))
            for s in steps if s.kind == "run"]
    left = [(label, n) for label, n in left if n]
    if left:
        print(f"===== queue {args.target} INCOMPLETE {_now()} -- relaunch to resume:",
              flush=True)
        for label, n in left:
            print(f"  {n:6d} fold-rows  {label}", flush=True)
        import tempfile
        free = disk.free_bytes(tempfile.gettempdir())
        if not disk.healthy(free):
            print(f"  the disk is below the {disk.gb(disk.HEALTHY_BYTES)} GB "
                  f"healthy reference ({disk.gb(free)} GB free): cells the disk "
                  f"rule left WAITING are among the rows above -- free space "
                  f"and relaunch", flush=True)
        raise SystemExit(1)
    print(f"===== queue {args.target} COMPLETE {_now()}", flush=True)


def main():
    from experiments.benchkit.scope import TARGETS
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--benchmark", required=True, choices=data.BENCHMARKS)
        sp.add_argument("--out", default=None,
                        help="CSV path (default results/bench_<benchmark>.csv)")

    def filters(sp):
        sp.add_argument("--arms", default="",
                        help="comma list (default: every arm declared on the "
                             "benchmark, in run order)")
        sp.add_argument("--datasets", default="",
                        help="comma list (default: all, smallest first)")
        sp.add_argument("--folds", default="",
                        help="comma list (default: the benchmark's folds)")

    def timing(sp):
        sp.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help="seconds per fold and of silence (default %(default)s)")
        sp.add_argument("--threads", type=int, default=DEFAULT_THREADS)
        sp.add_argument("--force", action="store_true",
                        help="start even if another timing driver is running")

    sp = sub.add_parser("queue", help="the whole run, from benchkit/scope.py")
    sp.add_argument("--target", default="all", choices=TARGETS)
    sp.add_argument("--dry-run", action="store_true",
                    help="print the steps and what each has left; run nothing")
    sp.add_argument("--results-dir", default=None,
                    help="where the result CSVs live (default results/)")
    timing(sp)
    sp.set_defaults(func=cmd_queue)

    sp = sub.add_parser("run")
    common(sp)
    filters(sp)
    timing(sp)
    sp.add_argument("--gate", action="store_true",
                    help="bank 'not run' rows for the gated cells (slow arms, "
                         "MONSTER, cost) predicted over the cap or the memory "
                         "budget (benchkit/gate.py)")
    sp.add_argument("--keep-cache", action="store_true",
                    help="do not drop a dataset's downloaded cache when the "
                         "plan is done with it (benchkit/reap.py)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("reap", help="drop the downloaded caches no unbanked "
                                     "cell reads any more")
    sp.add_argument("--results-dir", default=None)
    sp.add_argument("--dry-run", action="store_true",
                    help="print what would go, delete nothing")
    sp.set_defaults(func=cmd_reap)

    sp = sub.add_parser("plan")
    common(sp)
    filters(sp)
    sp.add_argument("--verbose", action="store_true",
                    help="list every cell, not just the per-arm counts")
    sp.add_argument("--gate", action="store_true",
                    help="also print what run --gate would decide for each "
                         "cell (feasibility gate, then disk rule); read-only")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("prefetch", help="download and cache Multiverse datasets")
    common(sp)
    sp.add_argument("--datasets", default="", help="comma list (default: all)")
    sp.add_argument("--arms", default="",
                    help="comma list: plan these arms' unbanked fold-0 cells "
                         "instead of the queue's scope")
    sp.add_argument("--needed-only", action="store_true",
                    help="of --datasets, fetch only what an unbanked planned "
                         "cell reads (the queue's prefetch steps pass it)")
    sp.set_defaults(func=cmd_prefetch)

    sp = sub.add_parser("child")
    sp.add_argument("--benchmark", required=True, choices=data.BENCHMARKS)
    sp.add_argument("--arm", required=True)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--folds", required=True)
    sp.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    sp.set_defaults(func=cmd_child)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
