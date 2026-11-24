"""One thread budget for numba and BLAS."""

import contextlib

import numba
import numpy as np

from .validation import _is_int


_TPC = []


def _blas_limit(k):
    """A context limiting BLAS to k threads (sklearn's threadpool
    controller), or None if that sklearn internal is unavailable."""
    if not _TPC:
        try:
            from sklearn.utils.parallel import _get_threadpool_controller
            _TPC.append(_get_threadpool_controller())
        except Exception:  # pragma: no cover - sklearn internals moved
            _TPC.append(None)
    if _TPC[0] is not None:
        try:
            return _TPC[0].limit(limits=int(k), user_api="blas")
        except Exception:  # pragma: no cover
            pass
    return None


def _ensure_omp():
    """Make numba pick its OpenMP layer, which the streaming head's team
    kernel needs (``_team_size`` is 1 under any other layer). numba fixes the
    layer at the process's first parallel region, and once aeon is imported
    its default priority picks tbb, which would run that head as a team of
    one (the same floats at a higher cost). While no layer is chosen,
    ``numba.config.THREADING_LAYER`` is set to "omp" for one tiny parallel
    call and restored after. Nothing happens when a layer is already chosen
    or when OpenMP is unavailable. ``numba.set_num_threads`` launches the
    layer too, so this runs before it."""
    try:
        numba.threading_layer()
        return
    except ValueError:
        pass
    from ..kernels.asgd import _asgd_finish
    # any float64 vector of the head's five scalars types the call as the
    # head's own (classification.stream._sgd_params): the values are unused
    sgd = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    prev = numba.config.THREADING_LAYER
    numba.config.THREADING_LAYER = "omp"
    try:
        _asgd_finish(np.zeros((2, 1, 2), dtype=np.float32),
                     np.ones(2, dtype=np.int64), np.int64(1), sgd,
                     np.empty((2, 1), dtype=np.float32))
    except Exception:            # OpenMP unavailable: numba's default layer
        pass
    finally:
        numba.config.THREADING_LAYER = prev


@contextlib.contextmanager
def _threads(k):
    """numba and BLAS at k threads for the block; yields whether BLAS was
    limited (False: BLAS runs at its own count, which fit records in
    fit_stats_['blas_limited'])."""
    _ensure_omp()
    prev = numba.get_num_threads()
    numba.set_num_threads(int(k))
    try:
        lim = _blas_limit(k)
        with (contextlib.nullcontext() if lim is None else lim):
            yield lim is not None
    finally:
        numba.set_num_threads(prev)


def _team_size():
    """Threads of a team kernel (kernels.team): the numba thread count under
    the omp layer, which runs prange(T) as T live threads; 1 under any other
    layer, or before any parallel region ran (tbb may run two iterations on
    one thread, and a spin barrier would then wait forever). A team of one
    computes the same floats."""
    try:
        layer = numba.threading_layer()
    except ValueError:
        return 1
    return int(numba.get_num_threads()) if layer == "omp" else 1


def _resolve_threads(n_jobs):
    """Threads for n_jobs: None is 1 (aeon's convention), -1 every numba
    thread, otherwise n_jobs capped at numba's thread count."""
    if n_jobs is None:
        return 1
    if not _is_int(n_jobs):
        raise ValueError(f"n_jobs must be None or an int, got {n_jobs!r}")
    n_jobs = int(n_jobs)
    top = int(numba.config.NUMBA_NUM_THREADS)
    if n_jobs == -1:
        return top
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be None, -1 or >= 1, got {n_jobs}")
    return min(n_jobs, top)
