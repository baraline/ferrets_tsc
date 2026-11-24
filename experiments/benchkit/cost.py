"""The cost benchmark's run discipline: which cells an arm may run, in
what order, and which rungs are not attempted once a lower rung failed.

Ladders are monotone in cost, so an arm that timed out, ran out of memory
or was refused by the gate at one rung is not attempted at the higher rungs
of the same axis; those cells are banked with the gate's prefix ("not run:
rung <cell> ...") so they read as predictions and deleting the rows forces
them. Anchors are independent of the grid and of each other.
"""

from . import synth
from .arms import is_univariate
from .data import sizes, split_kind
from .schema import SKIP_PREFIX, new_row

FAIL_PREFIXES = ("timeout", "start-up timeout", "MemoryError", "child failed")


def admissible(arm, dataset):
    return not (is_univariate(arm) and sizes("cost", dataset)[2] > 1)


def filter_cells(cells):
    return [c for c in cells
            if c.benchmark != "cost" or admissible(c.arm, c.dataset)]


def _rank(dataset):
    names = synth.grid()
    if dataset in names:
        return (0, names.index(dataset))
    n_train, n_test, C, m = sizes("cost", dataset)
    return (1, (n_train + n_test) * C * m, dataset)


def order(cells):
    """Arm-major in the arms' first-seen order; the grid in its order, then
    the anchors smallest first."""
    arms = list(dict.fromkeys(c.arm for c in cells))
    return sorted(cells, key=lambda c: (arms.index(c.arm), _rank(c.dataset)))


def failed(rows, timed_out=False):
    """A rung that timed out, ran out of memory or crashed the child. An
    estimator ValueError (e.g. a length floor) is a measured refusal, not a
    cost failure, and does not stop the axis."""
    if timed_out:
        return True
    return any(str(r.get("error") or "").startswith(FAIL_PREFIXES) for r in rows)


def pruned_by(cells, failed_cell):
    """The cells of the same arm above ``failed_cell`` on any axis it sits on."""
    if failed_cell.dataset not in synth.grid():
        return []
    axes = synth.axes_of(failed_cell.dataset)
    out = []
    for c in cells:
        if c.arm != failed_cell.arm or c.dataset not in synth.grid():
            continue
        above = synth.axes_of(c.dataset)
        if any(a in axes and above[a] > axes[a] for a in above):
            out.append(c)
    return out


def skip_rows(cell, failed_cell, reason, n_threads):
    return [new_row(cell.benchmark, cell.dataset, cell.arm, f,
                    split=split_kind(cell.benchmark, cell.dataset, f),
                    n_threads=int(n_threads),
                    error=f"{SKIP_PREFIX}rung {failed_cell.dataset} {reason}")
            for f in cell.folds]
