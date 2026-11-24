"""Datasets clean themselves once the run no longer needs them.

A full disk in the middle of a run turns cells into failures that are only
the full disk talking. Two caches this benchmark fills by itself carry the
weight:

* the Multiverse cache, one float64 ``.npz`` per dataset under
  ``~/aeon_data/_bench_npz`` -- 31.6 GB for the 115 problems MONSTER does
  not carry; the 18 it does carry add up to ~55 GB (S2Agri-17/34, which
  the disk rule excludes, left out). Caches are keyed (benchmark,
  dataset), so a problem's Multiverse .npz and its MONSTER hub repo are
  reaped independently;
* the MONSTER hub cache, ``~/.cache/huggingface/hub/datasets--monster-
  monash--<name>`` -- gigabytes per dataset.

Neither is needed once every planned cell that reads it is banked, and
both re-download, so the run drops them as it goes. The shape ``.json``
beside a Multiverse ``.npz`` stays: it is a few hundred bytes and the
feasibility gate and the cost order read sizes from it long
after the data itself is gone.

NEVER touched: the ``.ts`` archive under ``~/aeon_data`` (the local UCR
copy, which other scripts read and which no bench run created --
``data.prefetch`` already deletes only the ``.ts`` folders IT downloaded),
and any cache a still-unbanked cell needs, wherever in the plan it sits.
"""

import os
import shutil
from pathlib import Path

from .data import cache_file

MONSTER_REPO = "datasets--monster-monash--{name}"


def hub_cache():
    """The Hugging Face hub cache the MONSTER loader downloads into."""
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME")
    root = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return root / "hub"


def resolve(benchmark, dataset):
    """The (benchmark, dataset) whose cache this cell actually reads: a cost
    anchor is named ``<benchmark>:<dataset>`` and borrows another
    benchmark's data, so both names must mean the same cache to the keep
    set or an anchor would delete what its own benchmark still needs."""
    if benchmark == "cost" and ":" in dataset:
        inner, name = dataset.split(":", 1)
        return inner, name
    return benchmark, dataset


def cache_paths(benchmark, dataset):
    """Every path this benchmark downloaded for one dataset, biggest first.

    A cost anchor resolves to the cache of the benchmark it borrows from;
    a synthetic cell has none."""
    benchmark, dataset = resolve(benchmark, dataset)
    if benchmark == "cost":
        return []
    if benchmark == "multiverse":
        path = cache_file(dataset)
        return [path] if path.exists() else []
    if benchmark == "monster":
        repo = hub_cache() / MONSTER_REPO.format(name=dataset)
        return [repo] if repo.exists() else []
    return []                      # ucr: the local archive


def size_of(paths):
    """Bytes on disk. A hub repo's snapshots/ are SYMLINKS to its blobs/
    (where the box allows them), so links are skipped: following them
    counted every MONSTER cache twice."""
    total = 0
    for path in paths:
        if path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*")
                         if f.is_file() and not f.is_symlink())
        elif path.exists():
            total += path.stat().st_size
    return total


def reap(benchmark, dataset, keep=(), dry_run=False):
    """Drop one dataset's downloaded cache and return the bytes it held.

    ``keep`` is the set of ``(benchmark, dataset)`` pairs a cell somewhere
    in the plan still reads -- ``scope.still_needed()`` builds it. A pair
    inside it is left alone and 0 is returned, under either of its names:
    a cost anchor and its own benchmark share one cache."""
    keep = set(keep)
    if (benchmark, dataset) in keep or resolve(benchmark, dataset) in keep:
        return 0
    paths = cache_paths(benchmark, dataset)
    if not paths:
        return 0
    freed = size_of(paths)
    if dry_run:
        return freed
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                return 0
    return freed


def gb(n_bytes):
    return n_bytes / 2 ** 30
