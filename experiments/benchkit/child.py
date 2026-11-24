"""One cell = one (benchmark, dataset, arm), every fold, in one process.

Prints a line protocol on stdout and never touches the CSV:
    START <fold>     the fold's data are in place, the sampler runs
    ROW <json>       the fold's finished row (schema.COLUMNS) -- or, when
                     the disk does not hold the cell (ferrets's streaming
                     fit finds no room for its cache, STREAM_RECOMPUTE; or
                     loading or fitting runs out of space, out_of_space),
                     the fold's EXCLUSION row, error "not run: disk: ...",
                     and one such row for every later fold of the cell --
                     sent only when the fold STARTED on a healthy disk
                     (benchkit/disk.py healthy: the full disk is then the
                     dataset's doing)
    DISKWAIT <json>  the same, on a fold that started on a disk below the
                     healthy reference: the full disk may not be the
                     dataset's doing, so nothing is banked for this fold or
                     the later ones -- they stay for a relaunch once space
                     is freed ({"fold", "free", "reason"})
    DONE             the cell is over
Everything else on stdout is ignored by the parent.
"""

import datetime
import errno
import gc
import json
import pickle
import re
import warnings
from timeit import default_timer as timer

import numpy as np

from . import disk
from .arms import ARMS, build
from .data import load_folds, split_kind, warm_collection
from .memory import RssSampler, peak_wset_mb, rss_mb
from .schema import DISK_PREFIX, env_versions, new_row

CAUGHT = (ValueError, MemoryError, RuntimeError)
# ferrets's streaming fit sizes its cache from a 64-row probe BEFORE the
# transform pass and, when the cache would exceed STREAM_DISK_FRAC of the
# free temp disk, drops it and recomputes the transform every epoch (21
# passes instead of 1) with nothing but this RuntimeWarning to say so
# (ferrets.transform.chunks._cache_plan). Such a dataset makes the disk run out
# of space, and the protocol leaves it OUTSIDE the experiments, never running
# it in the recompute mode. The parent excludes,
# before the child starts, a cell whose predicted cache could not fit at
# any measured fill ratio (benchkit/disk.py); for a BORDERLINE cell, or a
# prediction that was too low, this is the BACKSTOP and ferrets's own
# probe the ground truth. The child turns the warning into an exception
# (it is raised after the probe, before any cache directory exists) and
# banks the fold -- and every later fold of the cell, on the same disk --
# as "not run: disk: ..." rows, the same exclusion the parent banks, so the
# queue still ends COMPLETE -- provided the fold STARTED on a healthy disk
# (disk.healthy); on a disk below the reference the probe's refusal may be
# the disk's state, not the dataset's size, and the fold is left unbanked
# (DISKWAIT) for a relaunch.
STREAM_RECOMPUTE = r".*recomputing the transform every epoch"


class _NoDisk(Exception):
    """ferrets found no room for its streaming cache (STREAM_RECOMPUTE)."""


def disk_error(warning):
    """The exclusion error of a fold whose streaming cache did not fit:
    ferrets's own numbers, without the recompute it would have done."""
    text = re.sub(r";\s*recomputing the transform every epoch\.?\s*$", "",
                  str(warning).strip())
    return (f"{DISK_PREFIX}: ferrets's own probe at fit time: {text} "
            f"(excluded; never run in recompute mode)")[:400]


# A fold that RUNS OUT of disk is no measurement either. The parent's disk
# rule makes it unlikely; this is the backstop for what it could not
# foresee. It is an exclusion only when the fold started on a healthy disk
# (a full disk is then the dataset's doing), and a DISKWAIT otherwise.
# Recognised by errno / winerror, or by the text the libraries put around it:
# the hub download's RuntimeError ("os error 112", localised: "Espace
# insuffisant sur le disque"), numpy's tofile short write ("requested and 0
# written"), Python's own ENOSPC text.
_NO_SPACE_ERRNO = {errno.ENOSPC}
_NO_SPACE_WINERROR = {39, 112}      # ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL
_NO_SPACE_TEXT = ("no space left on device", "os error 112",
                  "not enough space on the disk",
                  "espace insuffisant sur le disque", "disk full")
_SHORT_WRITE = re.compile(r"\d+ requested and \d+ written")


def out_of_space(error):
    """True when ``error`` -- or one it was raised from -- says the disk is
    full."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, OSError) and (
                error.errno in _NO_SPACE_ERRNO
                or getattr(error, "winerror", None) in _NO_SPACE_WINERROR):
            return True
        text = str(error).lower()
        if any(t in text for t in _NO_SPACE_TEXT) or _SHORT_WRITE.search(text):
            return True
        error = error.__cause__ or error.__context__
    return False


def full_disk_error(error, stage):
    """The exclusion error of a fold that ran out of disk space."""
    return (f"{DISK_PREFIX}: the disk ran out of space while {stage} "
            f"({type(error).__name__}: {str(error)[:200]}) (excluded; not a "
            f"measurement)")[:400]


def free_at_start(benchmark, dataset):
    """The least free space over the volumes the cell writes to, read at a
    fold's start (disk.cell_free); None when the disk cannot be read --
    which never counts as healthy."""
    try:
        return disk.cell_free(benchmark, dataset)
    except OSError:
        return None


def _disk_wait(write, fold, free, error):
    """Leave this fold and the later ones unbanked: the disk was below the
    healthy reference when the fold started, so its full disk is not
    evidence about the dataset."""
    write("DISKWAIT " + json.dumps(dict(
        fold=int(fold), free=None if free is None else float(free),
        reason=(f"{error[len(DISK_PREFIX):].lstrip(': ')} -- the fold started "
                f"with {'?' if free is None else disk.gb(free)} GB free, below "
                f"the {disk.gb(disk.HEALTHY_BYTES)} GB healthy reference: not "
                f"banked")[:400])))


def _excluded_rows(benchmark, dataset, arm, folds, n_threads, error):
    """One exclusion row per fold, for folds that never started."""
    return [new_row(benchmark, dataset, arm, int(f),
                    split=split_kind(benchmark, dataset, int(f)),
                    n_threads=int(n_threads), error=error) for f in folds]


def metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return dict(
        acc=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro",
                                zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted",
                                   zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(y_true, y_pred)),
    )


def _model_bytes(model):
    try:
        return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:  # noqa: BLE001 -- size is best-effort
        return None




def _warm_up(benchmark, arm, n_threads, dataset=None):
    """JIT and thread pools before any timer. An arm may bring its own
    warm-up (``Arm.warm``: ferrets crosses both ridge routes and its
    size-gated kernels, or both SGD schedules); the default is one fit and
    predict on the warm-up collection."""
    from .data import warm_channels
    X, y = warm_collection(benchmark,
                           n_channels=warm_channels(benchmark, dataset))
    spec = ARMS.get(arm)
    try:
        if spec is not None and spec.warm is not None:
            spec.warm(int(n_threads), X, y)
        else:
            build(arm, n_threads).fit(X, y).predict(X)
    except CAUGHT:
        pass  # the fold rows will carry the error; nothing to record here
    except Exception as error:  # noqa: BLE001 -- a full disk: the folds say so
        if not out_of_space(error):
            raise


def run_cell(benchmark, arm, dataset, folds, n_threads, write=None):
    write = write or (lambda line: print(line, flush=True))
    _warm_up(benchmark, arm, n_threads, dataset)
    base_wset = peak_wset_mb()
    versions = env_versions()
    device = "cpu"
    folds = [int(g) for g in folds]
    loader = load_folds(benchmark, dataset, folds)
    for k in range(len(folds)):
        # what the fold starts with: a full disk it meets is the dataset's
        # doing only if this was a healthy disk (disk.healthy)
        free0 = free_at_start(benchmark, dataset)
        try:
            f = next(loader)
        except StopIteration:
            break
        except Exception as error:  # noqa: BLE001 -- only a full disk is ours
            if not out_of_space(error):
                raise               # the child fails; the parent banks it
            # the download or the load ran out of disk: this fold and every
            # later one are excluded, none of them started -- or, on a disk
            # that was not healthy, left for a relaunch
            excluded = full_disk_error(error, "loading the data")
            if not disk.healthy(free0):
                _disk_wait(write, folds[k], free0, excluded)
                break
            for row in _excluded_rows(benchmark, dataset, arm, folds[k:],
                                      n_threads, excluded):
                write("ROW " + json.dumps(row))
            break
        gc.collect()
        base = rss_mb()
        row = new_row(
            benchmark, dataset, arm, f.fold,
            split=split_kind(benchmark, dataset, f.fold),
            n_train=int(len(f.y_train)), n_test=int(len(f.y_test)),
            n_channels=int(f.X_train.shape[1]), m=int(f.X_train.shape[2]),
            n_classes=int(f.n_classes), n_threads=int(n_threads),
            device=device, mem_base_mb=base,
            started_at=datetime.datetime.now().isoformat(timespec="seconds"),
            **versions)
        write(f"START {f.fold}")
        sampler = RssSampler().start()
        model = None
        excluded = None             # the exclusion error, when the disk ends it
        try:
            model = build(arm, n_threads)
            t0 = timer()
            no_disk = None
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=STREAM_RECOMPUTE,
                                        category=RuntimeWarning)
                try:
                    model.fit(f.X_train, f.y_train)
                except RuntimeWarning as warned:
                    if not re.match(STREAM_RECOMPUTE, str(warned)):
                        raise       # someone else's warning-as-error
                    no_disk = str(warned)
            if no_disk is not None:
                raise _NoDisk(no_disk)
            t1 = timer()
            pred = model.predict(f.X_test)
            t2 = timer()
            row.update(time_fit=t1 - t0, time_pred=t2 - t1,
                       **metrics(f.y_test, pred))
            n_features = getattr(model, "n_features_", None)
            if n_features is None and type(model).__module__.startswith(
                    "ferrets"):
                # the FERRETS classifier keeps the design's shape on its
                # fitted transformer
                n_features = getattr(getattr(model, "transformer_", None),
                                     "n_features_", None)
            row["n_features"] = (int(n_features) if n_features is not None
                                 else None)
        except _NoDisk as error:
            excluded = disk_error(str(error))
        except Exception as error:  # noqa: BLE001 -- re-raised unless ours
            if out_of_space(error):
                excluded = full_disk_error(error, "fitting or predicting")
            elif isinstance(error, CAUGHT):
                row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            else:
                raise
        finally:
            stats = sampler.stop()
        if excluded is not None and not disk.healthy(free0):
            # not evidence about the dataset: nothing banked, a relaunch
            # on a healthy disk runs this fold again
            _disk_wait(write, f.fold, free0, excluded)
            del model, f
            break
        if excluded is not None:
            # an exclusion, not a measurement: no time, memory or model
            for key in ("time_fit", "time_pred", "acc", "f1_macro",
                        "f1_weighted", "balanced_acc", "n_features"):
                row[key] = None
            row["error"] = excluded
            write("ROW " + json.dumps(row))
            # the same disk: every later fold of the cell is excluded too
            for later in _excluded_rows(benchmark, dataset, arm, folds[k + 1:],
                                        n_threads, excluded):
                write("ROW " + json.dumps(later))
            del model, f
            break
        row.update(mem_peak_mb=stats["peak_rss_mb"] - base,
                   mem_median_mb=stats["median_rss_mb"] - base,
                   mem_peak_rss_mb=stats["peak_rss_mb"],
                   mem_samples=stats["n_samples"])
        if model is not None and row["error"] is None:
            row["model_bytes"] = _model_bytes(model)  # outside the window
        row["mem_wset_peak_mb"] = peak_wset_mb() - base_wset
        write("ROW " + json.dumps(row))
        del model, f
    write("DONE")
