"""The plumbing the resample-10, memory and MUSE scripts share: one child
process per cell that prints a single ``ROWS <json>`` line, a wall-clock cap
that kills the child's whole process tree, and rows appended to a CSV
without rewriting the rows already there."""

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_child(argv, env, timeout=None):
    """Run one child; return (rows, None) from its ROWS line, or (None, the
    error text): "timeout after <timeout>s" when it overran ``timeout``
    seconds (None: no cap; the whole tree is killed, since on Windows
    killing the direct child leaves loky workers holding the pipe open), or
    "child failed (rc=...): <stderr tail>" when it printed no ROWS line."""
    from experiments.benchkit.parent import kill_tree
    child = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             cwd=REPO_ROOT, env=env)
    try:
        out, err = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(child)
        child.communicate()
        return None, f"timeout after {timeout}s"
    line = next((ln for ln in out.splitlines() if ln.startswith("ROWS ")), None)
    if line is None:
        tail = " | ".join(err.strip().splitlines()[-2:])
        return None, f"child failed (rc={child.returncode}): {tail[:400]}"
    return json.loads(line[len("ROWS "):]), None


def banked(path, key):
    """The set of ``key`` tuples (column names) of the rows in the CSV."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return set()
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    out = set()
    for row in df[list(key)].itertuples(index=False):
        out.add(tuple(_norm(v) for v in row))
    return out


def _norm(v):
    """A key field as text: 10.0 and 10 are the same thread count."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def key_of(row, key):
    return tuple(_norm(row.get(k)) for k in key)


def append_rows(path, rows, columns):
    """Append rows to the CSV, in its own header's column order (``columns``
    for a new file); the rows already there are not rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or not path.stat().st_size
    if new:
        header = list(columns)
    else:
        with open(path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) not in (b"\n", b"\r"):
                f.write(b"\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore",
                           lineterminator="\n")
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c))
                        for c in header})
