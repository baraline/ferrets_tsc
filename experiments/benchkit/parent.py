"""The parent side: which cells to run, one child per cell, the silence
clock that enforces the cap, tree kill, and atomic CSV banking.

Windows pitfalls this encodes:
  * ``subprocess.run(timeout=)`` kills only the direct child; joblib/loky
    workers survive holding the pipe. ``taskkill /T /F`` kills the tree
    (and ends a ``wsl.exe`` child's Linux process).
  * stdout AND stderr are drained on threads -- a child that logs more
    than the pipe buffer to stderr would otherwise block forever.
  * the venv ``python.exe`` is a launcher whose real interpreter is a
    child; ``/T`` covers it.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import pandas as pd

from .arms import ARMS
from .data import data_home, split_kind
from .schema import COLUMNS, new_row

REPO_ROOT = Path(__file__).resolve().parents[2]


def bank_path(benchmark, results_dir=None):
    """results/bench_<benchmark>.csv, or the same name under results_dir."""
    root = Path(results_dir) if results_dir else REPO_ROOT / "results"
    return root / f"bench_{benchmark}.csv"


THREAD_ENV = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS")
# MrSQM has no Windows wheel: its cells run in a separate interpreter under
# WSL2, with scikit-learn < 1.7 (MrSQM's own pin). FERRETS_MRSQM_PYTHON names
# that interpreter (a path inside the WSL distribution, e.g.
# ~/venv-mrsqm/bin/python), FERRETS_WSL_DISTRO the distribution (Ubuntu).
MRSQM_PYTHON_ENV = "FERRETS_MRSQM_PYTHON"
WSL_DISTRO_ENV = "FERRETS_WSL_DISTRO"
# the release's timing drivers: a bench run refuses to start while another
# one is alive (their timings would contaminate each other)
DRIVER_MARKERS = ("bench.py", "ablation.py", "seeds_ucr.py", "lsqr_exact.py",
                  "ties.py", "ucr_family_resample10.py",
                  "ucr_minirocket_resample10.py", "ucr_memory_runs.py",
                  "uea_muse.py")


# The temp-dir caches a streaming fit creates and removes only IN-PROCESS
# (ferrets's chunk cache closes on every exit path): a tree kill -- the cap,
# a paused queue -- skips that cleanup and leaves the cache on disk. A
# streaming cell needs room for its cache (disk only while the cache is <=
# half the free space -- ferrets's own test; benchkit/disk.py excludes a cell
# that would fail it), so orphans would silently shrink the room later
# stream cells get.
STREAM_PREFIXES = ("ferrets_stream_",)
# Caches that hold their files OPEN for their whole life: ferrets's chunk
# cache (ferrets/transform/chunk_cache.py) keeps its write handles until
# seal() and read handles from seal() to close(), so on Windows a live one's
# directory cannot be renamed and the rename is a liveness test.
HELD_OPEN_PREFIXES = ("ferrets_stream_",)
# an orphan sweep at start-up leaves a directory younger than this alone
# (a fit elsewhere between mkdtemp and opening its files)
ORPHAN_MIN_AGE_S = 120.0


def stream_dirs(root=None):
    """The stream-cache directories now in the temp dir (full paths)."""
    root = root or tempfile.gettempdir()
    try:
        return {e.path for e in os.scandir(root)
                if e.name.startswith(STREAM_PREFIXES) and e.is_dir()}
    except OSError:
        return set()


def remove_stream_dirs(paths, others_alive=None):
    """Remove these stream-cache directories, each only if it is provably
    dead.

    * ``ferrets_stream_`` caches (HELD_OPEN_PREFIXES): only if the directory
      can be renamed first. On Windows a directory holding a file open in any
      process cannot be renamed, and a live ferrets cache holds its files
      open, so a live fit's cache is kept.
    * any other prefix: the rename proves nothing, so it is removed only
      when ``others_alive`` is False -- no python process alive besides this
      one and its ancestors. ``others_alive`` None asks ``other_pythons()``
      (once, and only when such a directory is in ``paths``).

    Returns (removed, kept) basenames."""
    removed, kept = [], []
    for d in sorted(paths):
        name = os.path.basename(d)
        if not name.startswith(HELD_OPEN_PREFIXES):
            if others_alive is None:
                others_alive = bool(other_pythons())
            if others_alive:
                kept.append(name)
                continue
        tmp = d + ".reap"
        try:
            os.rename(d, tmp)
        except OSError:
            kept.append(name)
            continue
        shutil.rmtree(tmp, ignore_errors=True)
        removed.append(name)
    return removed, kept


def _older_than(paths, min_age_s, now):
    now = time.time() if now is None else now
    old = set()
    for d in paths:
        try:
            if now - os.stat(d).st_mtime >= min_age_s:
                old.add(d)
        except OSError:
            continue
    return old


def reap_orphan_stream_dirs(root=None, min_age_s=ORPHAN_MIN_AGE_S, now=None,
                            others_alive=None):
    """Remove the stream caches a killed run left behind (a paused queue
    kills the parent too, so run_child's own sweep never ran): every
    stream-cache directory older than ``min_age_s`` that remove_stream_dirs
    can prove dead. Call after the busy check. Returns (removed, kept)
    basenames."""
    return remove_stream_dirs(_older_than(stream_dirs(root), min_age_s, now),
                              others_alive=others_alive)


# joblib's memmapping folders (loky workers: BORF, MUSE_LR, ...) are named
# joblib_memmapping_folder_<pid of the process that made the pool>_<ids>
# and live in the temp dir on Windows. joblib removes them in-process; a
# tree kill (the cap, a paused queue) skips that, and the folder outlives
# its owner (several GB of them over a long run). Every GB left there is a
# GB less for the disk rule's allowance (benchkit/disk.py) -- whose
# exclusions are banked for good -- so the dead owners' folders go.
JOBLIB_PREFIX = "joblib_memmapping_folder_"


def joblib_dirs(root=None):
    """The joblib memmapping folders now in the temp dir (full paths)."""
    root = root or tempfile.gettempdir()
    try:
        return {e.path for e in os.scandir(root)
                if e.name.startswith(JOBLIB_PREFIX) and e.is_dir()}
    except OSError:
        return set()


def joblib_owner(path):
    """The pid in a joblib folder's name, or None."""
    head = os.path.basename(path)[len(JOBLIB_PREFIX):].split("_", 1)[0]
    return int(head) if head.isdigit() else None


def remove_joblib_orphans(paths, alive=None):
    """Remove the joblib folders whose OWNER process is dead and which can
    be renamed (on Windows a folder holding an open or mapped file cannot:
    a live pool's is kept even if its pid was misread). ``alive``: pid ->
    bool, default psutil.pid_exists. Returns (removed, kept) basenames."""
    if alive is None:
        import psutil
        alive = psutil.pid_exists
    removed, kept = [], []
    for d in sorted(paths):
        name = os.path.basename(d)
        pid = joblib_owner(d)
        if pid is None or alive(pid):
            kept.append(name)
            continue
        tmp = d + ".reap"
        try:
            os.rename(d, tmp)
        except OSError:
            kept.append(name)
            continue
        shutil.rmtree(tmp, ignore_errors=True)
        removed.append(name)
    return removed, kept


def reap_orphan_joblib_dirs(root=None, min_age_s=ORPHAN_MIN_AGE_S, now=None,
                            alive=None):
    """Remove the joblib folders killed processes left behind: older than
    ``min_age_s``, owner dead, renameable. Returns (removed, kept)."""
    return remove_joblib_orphans(_older_than(joblib_dirs(root), min_age_s, now),
                                 alive=alive)


@dataclass(frozen=True)
class Cell:
    benchmark: str
    dataset: str
    arm: str
    folds: tuple


def plan_cells(benchmark, arms, datasets, folds, banked):
    """Dataset-major, arms in the given order, only the unbanked folds.

    ``folds`` is either one sequence for every arm or a {arm: sequence}
    mapping -- the slow arms run fewer folds than the rest."""
    cells = []
    for dataset in datasets:
        for arm in arms:
            want = folds[arm] if isinstance(folds, dict) else folds
            missing = tuple(int(f) for f in want
                            if (dataset, arm, int(f)) not in banked)
            if missing:
                cells.append(Cell(benchmark, dataset, arm, missing))
    return cells


class CsvBank:
    """Rows keyed (dataset, model, fold); every add rewrites the file
    atomically (temp file + os.replace) so a kill never truncates it."""

    def __init__(self, path):
        self.path = Path(path)
        self.frames = []
        if self.path.exists():
            previous = pd.read_csv(self.path)
            if len(previous):
                self.frames.append(previous)

    def frame(self):
        if not self.frames:
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat(self.frames, ignore_index=True).reindex(columns=COLUMNS)

    def keys(self):
        df = self.frame()
        if not len(df):
            return set()
        return set(zip(df["dataset"], df["model"], df["fold"].astype(int)))

    def add(self, rows):
        if not rows:
            return
        self.frames.append(pd.DataFrame(rows))
        # one frame, not one per banked row: concatenating thousands of
        # one-row frames on every add made banking quadratic in the run
        self.frames = [self.frame()]
        self.write()

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".",
                                   suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        try:
            self.frame().to_csv(tmp, index=False)
            # Windows refuses the replace while another process has the CSV
            # open (a reader, an indexer): retry briefly. A failure still
            # loses no row -- they stay in memory for the next write -- but
            # its temp file must not be left behind.
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.2 * (attempt + 1))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def child_env(n_threads):
    env = dict(os.environ)
    for var in THREAD_ENV:
        env[var] = str(int(n_threads))
    return env


def _wsl_path(path):
    p = Path(path).resolve()
    drive = p.drive.rstrip(":").lower()
    return "/mnt/" + drive + p.as_posix()[len(p.drive):]


def interpreter_missing(arm):
    """The reason a separate-interpreter arm cannot run here, or None: on
    Windows an arm flagged ``wsl`` (MrSQM) needs FERRETS_MRSQM_PYTHON. Its
    cells stay unbanked, so the queue ends INCOMPLETE until it is set."""
    if (ARMS[arm].wsl and sys.platform == "win32"
            and not os.environ.get(MRSQM_PYTHON_ENV)):
        return f"not run: {arm} interpreter not configured ({MRSQM_PYTHON_ENV})"
    return None


def _hf_home():
    """The Hugging Face cache the Windows side uses, as a WSL path."""
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return _wsl_path(home)


def wsl_command(script, args, n_threads):
    """argv running ``experiments/<script> <args>`` with the
    FERRETS_MRSQM_PYTHON interpreter inside WSL (FERRETS_WSL_DISTRO, default
    Ubuntu), the thread caps, the Windows side's aeon data home and Hugging
    Face cache exported, from the repository's WSL path: both sides read the
    same downloads."""
    interpreter = os.environ[MRSQM_PYTHON_ENV]
    distro = os.environ.get(WSL_DISTRO_ENV, "Ubuntu")
    exports = " ".join(f"export {v}={int(n_threads)};" for v in THREAD_ENV)
    inner = (f"{exports} export AEON_DATA={shlex.quote(_wsl_path(data_home()))}; "
             f"export HF_HOME={shlex.quote(_hf_home())}; "
             f"cd {shlex.quote(_wsl_path(REPO_ROOT))} && {interpreter} "
             f"experiments/{script} " + " ".join(shlex.quote(a) for a in args))
    return ["wsl.exe", "-d", distro, "--", "bash", "-lc", inner]


def child_command(cell, n_threads, python=None):
    """The child's argv; the MrSQM arm goes through wsl.exe on Windows."""
    args = ["child", "--benchmark", cell.benchmark, "--arm", cell.arm,
            "--dataset", cell.dataset,
            "--folds", ",".join(str(f) for f in cell.folds),
            "--threads", str(int(n_threads))]
    script = REPO_ROOT / "experiments" / "bench.py"
    if ARMS[cell.arm].wsl and sys.platform == "win32":
        return wsl_command("bench.py", args, n_threads)
    return [python or sys.executable, str(script)] + args


def kill_tree(proc):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        import psutil
        try:
            for c in psutil.Process(proc.pid).children(recursive=True):
                c.kill()
        except psutil.NoSuchProcess:
            pass
        proc.kill()


@dataclass
class ChildResult:
    rows: list
    started: list
    done: bool
    rc: object
    timed_out: bool
    stderr_tail: str
    # stream caches removed after the child was killed or crashed
    stream_dirs_removed: tuple = ()
    # joblib memmapping folders of its dead processes, removed likewise
    joblib_dirs_removed: tuple = ()
    # the child's DISKWAIT: a fold met a full disk, or its streaming cache
    # found no room, on a disk that was below the healthy reference when it
    # started -- its reason; that fold and the later ones stay UNBANKED
    disk_wait: str = None


def run_child(cmd, env, cap_s, cwd=REPO_ROOT, on_row=None, poll_s=0.5,
              sweep=True):
    """Run one child; kill it when no protocol line arrives for cap_s.

    A child that was killed or exited non-zero cannot have removed its own
    stream cache or its joblib memmapping folders: with ``sweep`` the ones
    that appeared during its window are removed afterwards
    (remove_stream_dirs: a cache another live process may own is kept;
    remove_joblib_orphans: only a dead owner's). bench.py passes
    ``sweep=False`` under --force, which may run beside other drivers whose
    caches could appear in the same window."""
    before = stream_dirs() if sweep else set()
    before_jl = joblib_dirs() if sweep else set()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(cwd), env=env)
    state = dict(last=timer(), rows=[], started=[], done=False, disk_wait=None)
    lock = threading.Lock()
    err_lines = deque(maxlen=50)

    def read_out():
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            with lock:
                if line.startswith("START "):
                    state["started"].append(int(line.split()[1]))
                    state["last"] = timer()
                elif line.startswith("ROW "):
                    row = json.loads(line[4:])
                    state["rows"].append(row)
                    state["last"] = timer()
                    if on_row is not None:
                        try:
                            on_row(row)
                        except Exception as error:  # noqa: BLE001
                            err_lines.append(f"on_row failed: {error!r}")
                elif line.startswith("DISKWAIT "):
                    try:
                        info = json.loads(line[9:])
                        state["disk_wait"] = str(info.get("reason") or "disk")
                    except ValueError:
                        state["disk_wait"] = line[9:][:400] or "disk"
                    state["last"] = timer()
                elif line.strip() == "DONE":
                    state["done"] = True
                    state["last"] = timer()

    def read_err():
        for line in proc.stderr:
            err_lines.append(line.rstrip("\r\n"))

    threads = [threading.Thread(target=read_out, daemon=True),
               threading.Thread(target=read_err, daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    while proc.poll() is None:
        with lock:
            silent = timer() - state["last"]
        if cap_s is not None and silent > cap_s:
            timed_out = True
            kill_tree(proc)
            break
        time.sleep(poll_s)
    proc.wait()
    # The child may have exited with ROW/DONE lines still unread or still
    # being banked by on_row. Wait for the stdout reader as long as it makes
    # progress: reading state before it drains would bank a finished fold as
    # "child failed (rc=0)" and then the real row beside it. Only a reader
    # that is silent for 10 s (an orphan holding the pipe) is abandoned.
    out_reader, err_reader = threads
    while True:
        with lock:
            seen = state["last"]
        out_reader.join(timeout=10)
        if not out_reader.is_alive():
            break
        with lock:
            if state["last"] == seen:
                break
    err_reader.join(timeout=10)
    removed = removed_jl = ()
    if sweep and (timed_out or proc.returncode != 0):
        removed = tuple(remove_stream_dirs(stream_dirs() - before)[0])
        removed_jl = tuple(remove_joblib_orphans(joblib_dirs() - before_jl)[0])
    tail = " | ".join(ln for ln in list(err_lines)[-3:] if ln.strip())[:400]
    return ChildResult(rows=state["rows"], started=state["started"],
                       done=state["done"], rc=proc.returncode,
                       timed_out=timed_out, stderr_tail=tail,
                       stream_dirs_removed=removed,
                       joblib_dirs_removed=removed_jl,
                       disk_wait=state["disk_wait"])


def outcome_rows(cell, result, n_threads, cap_s):
    """Rows the parent must add for folds the child produced no ROW for.

    Every fold ends banked: measured, failed, timed out, not attempted, or
    excluded ("not run: disk", which the child sends as an ordinary ROW --
    child.STREAM_RECOMPUTE). The one exception is a DISKWAIT: the disk was
    below the healthy reference when the fold started (benchkit/disk.py),
    so its full disk says nothing about the dataset, and the folds the child
    sent no row for stay unbanked for a relaunch on a healthy disk."""
    if result.disk_wait is not None:
        return []
    have = {int(r["fold"]) for r in result.rows}
    pending = [f for f in cell.folds if f not in have]
    if not pending:
        return []

    def row(fold, error):
        return new_row(cell.benchmark, cell.dataset, cell.arm, fold,
                       split=split_kind(cell.benchmark, cell.dataset, fold),
                       n_threads=int(n_threads), error=error)
    clean = result.done and result.rc == 0 and not result.timed_out
    if clean:
        return [row(f, "child finished without a row for this fold")
                for f in pending]
    in_flight = pending[0]
    if result.timed_out:
        if result.started and result.started[-1] == in_flight:
            first = f"timeout after {cap_s}s"
        else:
            first = f"start-up timeout after {cap_s}s"
        reason = "timed out"
    else:
        first = f"child failed (rc={result.rc}): {result.stderr_tail}"
        reason = "failed"
    rows = [row(in_flight, first)]
    rows += [row(f, f"not attempted: fold {in_flight} {reason}")
             for f in pending[1:]]
    return rows


def _python_processes():
    """(pid, cmdline) of the live python processes other than this one and
    EVERY ancestor: ``bench.py queue`` runs each step as a ``bench.py run``
    child, and the venv's python.exe is a launcher whose real interpreter
    is its child."""
    import psutil
    me = psutil.Process()
    skip = {me.pid, os.getppid()}
    try:
        skip |= {p.pid for p in me.parents()}
    except psutil.Error:
        pass
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] in skip:
            continue
        name = (p.info["name"] or "").lower()
        if "python" in name:
            out.append((p.info["pid"], " ".join(p.info["cmdline"] or [])))
    return out


def other_pythons():
    """Every other live python process (see _python_processes), as
    "pid: cmdline" strings: a cache that holds no file open may belong to
    any of them."""
    return [f"{pid}: {cmd[:120]}" for pid, cmd in _python_processes()]


def busy_processes(markers=None):
    """Other timing drivers alive on the box (their timings and ours would
    contaminate each other): python processes whose command line carries
    one of ``markers`` (default DRIVER_MARKERS), this process and its
    ancestors excluded (see _python_processes)."""
    markers = set(DRIVER_MARKERS if markers is None else markers)
    return [f"{pid}: {cmd[:120]}" for pid, cmd in _python_processes()
            if markers & _script_names(cmd)]


def _script_names(cmd):
    """The file names of a command line's tokens: a driver is matched by its
    script's name, not by a substring (test_bench.py is not bench.py)."""
    return {tok.replace("\\", "/").rsplit("/", 1)[-1].strip('"')
            for tok in cmd.split()}
