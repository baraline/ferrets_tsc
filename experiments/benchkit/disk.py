"""The disk rule: a dataset that would make the disk run out of space is left
outside the experiments.

A cell is EXCLUDED -- banked once, one row per fold, with an error starting
``not run: disk`` (schema.DISK_PREFIX: of the gate's SKIP_PREFIX family, so
it is a prediction and never a measured failure; the paper states these
datasets as excluded) -- when running it would
exhaust the disk:

1. its DATA: the bytes the cell must download before it can run -- a
   MONSTER dataset's hub files (data.MONSTER_HUB_BYTES, exact), a
   Multiverse dataset whose .npz cache is gone -- exceed the free space
   above the queue's floor (DISK_FLOOR_GB, the free space under which
   bench.py refuses to start at all);
2. its STREAMING CACHE (ferrets's streaming arm only, STREAM_ARMS):
   ferrets writes the cache only while its estimate is at most
   STREAM_DISK_FRAC of the free temp disk (ferrets.transform.chunks._cache_plan;
   otherwise it recomputes the transform every epoch, which the protocol
   does not time), and the cache must also leave the floor free. So the
   ALLOWANCE, with ``room`` the free temp disk once the data are in, is
   min(STREAM_DISK_FRAC x room, room - floor): never more than ferrets
   itself would accept, and no stricter than the two conditions together.

Decided in the PARENT just before each cell (bench.py; also before each
download in ``prefetch``), against the free disk of that moment, so an
excluded dataset is never downloaded. The data size is known exactly; the
cache is PREDICTED as ferrets would estimate it from its 64-row probe --
STREAM_EST_BYTES x the probe's entries per row x n_train -- with the entries
per row predicted as a fill ratio x ``expected_keys``: every block of the
arm's own draw, holding W windows over a keyspace of KS keys, would show
KS (1 - (1 - 1/KS)^W) distinct keys if the keys were uniform. The blocks
follow the arm's cache GEOMETRY (STREAM_GEOMETRY):

* separable (ferrets at C >= 2): per side (the channels, their first
  differences), the family draw (ferrets.transform.views._draw_families); per
  distinct (family, dilation), k channel blocks of 256 keys, then -- when k > 1
  -- the coarse word (256 keys, meta bit 0), the change flags (2^k) and the
  agreement flags (2^(k-1)) (meta bit 1);
* per_channel (C = 1, and the ``*_perchannel`` arms): per pseudo-channel
  (every channel and its first difference), every distinct (dictionary,
  window, dilation) block of the arm's own view draw, of 256 keys;
* with the level word (the third element of the geometry): one more
  256-key block
  per distinct (pseudo-channel or side, channels, window, dilation),
  deduplicated over the sources (the first view keying it owns it). The fill
  ratios below were measured without it; for level blocks the uniform model
  is a bound until a level-on probe measures them.

The two coincide at C = 1. That geometry matters: at W ~ KEYSPACE (m ~
256, the DREAMER sets) uniform keys fill only ~63% of the block, where the
plain min(KS, W) bound (``keyspace_bound``) would say 100%; and at small m
(S2Agri-10pc, m = 24) the per-channel geometry under-predicts the separable
cache several-fold. The measured probe-to-expected ratio over seven timed
per-channel MONSTER streaming cells (entries per row = STREAM_CHUNK_NNZ /
chunk_rows) is 0.478 LakeIce,
0.534 InsectSound, 0.725 Skoda, 0.805 AudioMNIST-DS, 0.817 FruitFlies, 0.832
WISDM, 0.895 CornellWhaleChallenge: FILL_RATIO is the median, FILL_RATIO_LOW
/ _HIGH the extremes. For the separable geometry the bracket is widened by
ferrets's own separable probes (SEP_PROBES: the estimates it printed on the
DREAMER and S2Agri-10pc sets), derived when first needed (``fill_ratios``); the median stays FILL_RATIO.

The ratio depends on the data (entropy), so the parent EXCLUDES a cell only
when its cache would not fit even at the lowest fill ratio of its geometry
(FILL_RATIO_LOW, or below it for the separable one) -- a cell that cannot
run whatever its data look like. Between the two (``borderline``) the cell
runs and ferrets's own probe -- the ground truth -- decides at fit time:
the child is the BACKSTOP, banking the same ``not run: disk`` row
(child.STREAM_RECOMPUTE) instead of fitting when the probe finds no room,
and also when a fold actually runs out of space (child.out_of_space). The
price of a borderline cell is its download before the probe decides.

A HEALTHY DISK: an exclusion is banked for good, so it must say something
about the DATASET, not about how full the disk happens to be that hour. So
every verdict is read against max(free now, DISK_HEALTHY_GB): a cell is
EXCLUDED only when it would not fit even then. A cell that would fit a
healthy disk but not the disk as it is now WAITS (``Verdict.wait``): nothing
is banked, the run moves on to the next cell, and the queue ends INCOMPLETE
until space is freed and it is relaunched. On a disk below the reference a
borderline cell waits too (its probe's refusal could not be told from the
disk's state), and the child banks an exclusion of its own only when the
fold STARTED on a healthy disk (``cell_free`` / ``healthy``; otherwise it
sends DISKWAIT and the cell stays unbanked). DISK_HEALTHY_GB is the free
space of the measuring machine in its normal state (60 to 76 GB), not a
fitted constant.

All sizes are in GB = 1e9 bytes, as ferrets's own warning prints them;
the floor is DISK_FLOOR_GB GiB (bench.py's unit).
"""

import functools
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arms import FERRETS_PARAMS
from .data import (MONSTER_HUB_BYTES, MONSTER_SHAPES, cache_file, data_home,
                   datasets, multiverse_overlap, sizes, split_kind)
from .reap import MONSTER_REPO, cache_paths, hub_cache, resolve, size_of
from .schema import DISK_PREFIX, new_row

# bench.py refuses to start below this much free space (GiB); the disk rule
# keeps every cell above it.
DISK_FLOOR_GB = 10.0
FLOOR_BYTES = DISK_FLOOR_GB * 2 ** 30
# ferrets's streaming constants, mirrored so the parent never imports
# ferrets (numba).
STREAM_EST_BYTES = 2.4      # cache bytes per probe entry (its pre-flight estimate)
STREAM_DISK_FRAC = 0.5      # the cache must fit in this share of the free disk
STREAM_RAM_MB = 256.0       # a cache up to this size stays in RAM: no disk
STREAM_CHUNK_NNZ = 2.5e7    # entries per chunk: chunk_rows = this // per row
# the free disk of the machine in its normal state (module docstring): a
# verdict is read against max(free now, this), so a disk below it can make a
# cell WAIT but never gets it excluded for good
DISK_HEALTHY_GB = 60.0
HEALTHY_BYTES = DISK_HEALTHY_GB * 1e9
KEYSPACE = 256              # keys per (dictionary, dilation) block: 2^8 words
# the arms whose fit is ferrets's streaming fit, and the geometry of their
# cache: (layout, meta code -- bit 0 the coarse word, bit 1 the flags --,
# level word)
STREAM_GEOMETRY = {
    "FERRETS_stream": ("separable", 3, True),
    "FERRETS_stream_unweighted": ("separable", 3, True),
    "FERRETS_stream_perchannel": ("per_channel", None, True),
}
STREAM_ARMS = frozenset(STREAM_GEOMETRY)
# ferrets's own probe on the separable design: (dataset, meta) -> (the cache
# estimate it printed, GB; n_train of the fold-0 split)
SEP_PROBES = {("DREAMERA", 3): (31.9, 140638), ("DREAMERV", 3): (31.9, 140638),
              ("S2Agri-10pc-17", 3): (86.7, 4661858),
              ("S2Agri-10pc-34", 3): (86.7, 4661858),
              ("S2Agri-10pc-17", 2): (76.0, 4661858),
              ("S2Agri-10pc-34", 2): (76.0, 4661858)}
# the family draw's constants (ferrets.constants._SUBSET_SALT, _FAMILY_KS,
# _SOURCES), mirrored so the parent never imports ferrets
SUBSET_SALT = 1000003
FAMILY_KS = (2, 4, 8)
SOURCES = ("paa", "fourier")
# measured probe entries per row / expected_keys, per-channel geometry
# (module docstring)
FILL_RATIO = 0.805          # median of the seven
FILL_RATIO_LOW = 0.478      # LakeIce: the parent excludes only past this
FILL_RATIO_HIGH = 0.895     # CornellWhaleChallenge: fits for sure under this
# a Multiverse dataset that is not cached is downloaded as .ts text (aeon,
# from Zenodo) and cached as float64 .npz before the .ts is deleted
# (data.prefetch): the peak holds both. Measured .ts / .npz on local copies
# of UEA sets: PEMS-SF 0.86, FaceDetection 1.17, LSST 0.94 --
# so 1 + ~1.2, rounded up for the archive the download arrives in.
MV_DOWNLOAD_FACTOR = 2.5
# The problems MONSTER also carries (data.multiverse_overlap): no .ts of
# theirs has been measured. ASSUMED: values written at up to 17 significant
# digits (~20 characters, ~2.5x the float64 bytes), so .ts + .npz ~3.5x.
# aeon's download also holds the zip beside the .ts until extraction ends,
# so 3.5 is a bound only while the zip is under the .npz. data.prefetch
# records each .ts's bytes (``ts_bytes``) and ``bench.py prefetch`` prints
# its ratio to the .npz: replace this constant by that measurement.
MV_OVERLAP_DOWNLOAD_FACTOR = 3.5
# a WHAT-IF for previews only: free GB to assume instead of reading the
# disk. Read by ``bench.py queue --dry-run`` (whatif_free); a run, a queue
# and prefetch always read the real disk
# (bench.py says so when the variable is set).
FREE_ENV = "BENCH_FREE_DISK_GB"


@dataclass(frozen=True)
class Verdict:
    run: bool
    reason: str                 # what follows "not run: disk: " when excluded
    download: float = 0.0       # bytes the cell downloads first
    cache: float = 0.0          # ferrets's predicted cache estimate (median fill)
    cache_low: float = 0.0      # ... at the geometry's lowest fill ratio
    cache_high: float = 0.0     # ... at its highest (fill_ratios)
    allowance: float = None     # bytes the cache may take (disk mode only)
    free: float = None          # bytes free on the volume that binds
    borderline: bool = False    # runs; ferrets's own probe decides
    wait: bool = False          # not run AND not banked: the disk is below
                                # the healthy reference and the cell would
                                # fit a healthy one (module docstring)

    @property
    def excluded(self):
        """Banked for good as "not run: disk"."""
        return not self.run and not self.wait


def gb(n_bytes):
    return f"{n_bytes / 1e9:,.1f}"


# ------------------------------------------------------------- the cache ---

@functools.lru_cache(maxsize=None)
def _pc_windows(p, m, level=False):
    """Windows per distinct block of pseudo-channel ``p`` of an m-point
    series in the per_channel geometry: ferrets's per-pseudo-channel view
    draw for it (RandomState(seed * 7919 + p): source, window, one
    log-uniform dilation, ``n_dictionaries`` times -- the statements of
    ferrets.transform.views._draw_views_py, which ferrets.FERRETS._draw_views
    replays at C = 1 and under channel_layout="per_channel"), duplicates
    collapsed; a block of
    window w at dilation d slides over m - (w - 1) d positions. With level, one
    level block per distinct (window, dilation) follows, deduplicated over the
    sources."""
    windows = [int(w) for w in FERRETS_PARAMS["windows"] if int(w) <= m]
    if not windows:
        return ()
    seed = int(FERRETS_PARAMS["random_state"])
    rng = np.random.RandomState((seed * 7919 + p) % (1 << 32))
    blocks = set()
    for _ in range(int(FERRETS_PARAMS["n_dictionaries"])):
        source = rng.randint(2)
        w = windows[rng.randint(len(windows))]
        max_d = max(1, (m - 1) // (w - 1))
        d = int(2 ** rng.uniform(0.0, np.log2(max_d + 1)))
        blocks.add((source, w, max(1, min(d, max_d))))
    lev = sorted({(w, d) for _, w, d in blocks}) if level else []
    return (tuple(m - (w - 1) * d for _, w, d in sorted(blocks))
            + tuple(m - (w - 1) * d for w, d in lev))


@functools.lru_cache(maxsize=None)
def _family_blocks(p, m, C, meta, level=False):
    """(windows, keyspace) per block of side ``p`` of an m-point, C-channel
    series in the separable geometry: ferrets's family draw for it
    (ferrets.transform.views._draw_families, statement for statement:
    RandomState(seed * 7919 + p) draws source, window and one log-uniform
    dilation, RandomState(seed * 7919 + SUBSET_SALT + p) draws k and the
    channels, ``n_dictionaries`` times; families sharing (source, window,
    channels) merge their dilations). Per (family, dilation), in the kernel's
    block order: k channel blocks of KEYSPACE keys, then, when k > 1, the
    coarse word
    (KEYSPACE, ``meta`` bit 0) and the change (2^k) and agreement (2^(k-1))
    flags (bit 1). With level, one 256-key level block per distinct (window,
    channels, dilation), deduplicated over the sources, after them all. Empty
    when no window fits; never raises."""
    windows = [int(w) for w in FERRETS_PARAMS["windows"] if int(w) <= m]
    if not windows:
        return ()
    seed = int(FERRETS_PARAMS["random_state"])
    ks = [k for k in FAMILY_KS if k <= C] or [1]
    rng = np.random.RandomState((seed * 7919 + p) % (1 << 32))
    srng = np.random.RandomState((seed * 7919 + SUBSET_SALT + p) % (1 << 32))
    draws = {}
    for _ in range(int(FERRETS_PARAMS["n_dictionaries"])):
        source = SOURCES[rng.randint(len(SOURCES))]
        w = windows[rng.randint(len(windows))]
        max_d = max(1, (m - 1) // (w - 1))
        d = int(2 ** rng.uniform(0.0, np.log2(max_d + 1)))
        d = max(1, min(d, max_d))
        k = ks[srng.randint(len(ks))]
        chs = tuple(sorted(int(c) for c in srng.choice(C, k, replace=False)))
        draws.setdefault((source, w, chs), set()).add(d)
    blocks = []
    for (_, w, chs), ds in sorted(draws.items()):
        k = len(chs)
        for d in sorted(ds):
            W = m - (w - 1) * d
            if W <= 0:
                continue
            blocks += [(W, KEYSPACE)] * k
            if k > 1:
                if meta & 1:
                    blocks.append((W, KEYSPACE))
                if meta & 2:
                    blocks += [(W, 2 ** k), (W, 2 ** (k - 1))]
    if level:
        for w, chs, d in sorted({(w, chs, d) for (_, w, chs), ds in
                                 draws.items() for d in ds}):
            W = m - (w - 1) * d
            if W > 0:
                blocks.append((W, KEYSPACE))
    return tuple(blocks)


def _blocks(m, n_channels, design, meta, level=False):
    """(windows, keyspace) of every block of a row, over the pseudo-channels
    (per_channel) or the two sides (separable); with level, the level
    blocks too."""
    m, C = int(m), int(n_channels)
    if design == "per_channel" or C == 1:
        return [(W, KEYSPACE) for p in range(2 * C)
                for W in _pc_windows(p, m, bool(level))]
    if design != "separable":
        raise ValueError(f"unknown cache design {design!r}")
    return [b for p in (0, 1)
            for b in _family_blocks(p, m, C, int(meta), bool(level))]


def keyspace_bound(m, n_channels, design="separable", meta=3, level=False):
    """Distinct keys per row at most: every block holds at most
    min(its keyspace, its windows)."""
    return sum(min(ks, W)
               for W, ks in _blocks(m, n_channels, design, meta, level))


def expected_keys(m, n_channels, design="separable", meta=3, level=False):
    """Distinct keys per row if every block's keys were uniform over its
    keyspace: KS (1 - (1 - 1/KS)^W) per block of W windows."""
    return float(sum(ks * (1.0 - (1.0 - 1.0 / ks) ** W)
                     for W, ks in _blocks(m, n_channels, design, meta,
                                          level)))


def _design(arm, n_channels):
    """The cache geometry of a streaming arm's cell: (layout, meta, level);
    one channel is the per_channel layout, with the arm's level flag."""
    layout, meta, level = STREAM_GEOMETRY[arm]
    if int(n_channels) == 1:
        return ("per_channel", None, level)
    return (layout, meta, level)


@functools.lru_cache(maxsize=None)
def _sep_probe_ratios():
    """{(dataset, meta): probe entries per row / expected_keys} of ferrets's
    own separable probes (SEP_PROBES)."""
    out = {}
    for (name, meta), (gb_est, n_train) in SEP_PROBES.items():
        _, C, m = MONSTER_SHAPES[name]
        per_row = gb_est * 1e9 / (STREAM_EST_BYTES * n_train)
        out[(name, meta)] = per_row / expected_keys(m, C, "separable", meta)
    return out


def fill_ratios(design):
    """(low, median, high) fill ratios of a cache geometry: the seven
    per-channel probes; for the separable geometry the bracket widened by
    the separable probes (module docstring)."""
    if design == "per_channel":
        return (FILL_RATIO_LOW, FILL_RATIO, FILL_RATIO_HIGH)
    r = list(_sep_probe_ratios().values())
    return (min(FILL_RATIO_LOW, *r), FILL_RATIO, max(FILL_RATIO_HIGH, *r))


def stream_cache_bytes(benchmark, arm, dataset, ratio=None):
    """ferrets's pre-flight cache estimate for this cell, predicted at a
    fill ratio (default the median of the cell's geometry); 0 for an arm
    without a streaming cache."""
    if arm not in STREAM_GEOMETRY:
        return 0.0
    n_train, _, n_channels, m = sizes(benchmark, dataset)
    design, meta, level = _design(arm, n_channels)
    if ratio is None:
        ratio = fill_ratios(design)[1]
    # meta is None only for the per_channel layout, where _blocks ignores
    # it; meta 0 (no meta-words) is a real code and passes through
    return (STREAM_EST_BYTES * float(ratio)
            * expected_keys(m, n_channels, design, 3 if meta is None else meta,
                            level)
            * float(n_train))


# -------------------------------------------------------------- the data ---

def _hub_has(name):
    repo = hub_cache() / MONSTER_REPO.format(name=name)
    return any(repo.glob(f"snapshots/*/{name}_X.npy"))


def download_bytes(benchmark, dataset):
    """Bytes the cell downloads before it can run: 0 for the box's own
    archive (UCR), a generated cell or a dataset already cached."""
    bench, name = resolve(benchmark, dataset)
    if bench == "monster":
        if _hub_has(name):
            return 0.0
        return resident_bytes(bench, name)
    if bench == "multiverse":
        if cache_file(name).exists():
            return 0.0
        factor = (MV_OVERLAP_DOWNLOAD_FACTOR if name in _overlap()
                  else MV_DOWNLOAD_FACTOR)
        return factor * resident_bytes(bench, name)
    return 0.0


@functools.lru_cache(maxsize=None)
def _overlap():
    return frozenset(multiverse_overlap())


def resident_bytes(benchmark, dataset):
    """What a dataset's cache holds on disk once downloaded: a MONSTER
    dataset's hub files (exact), a Multiverse dataset's float64 .npz."""
    if benchmark == "monster":
        exact = MONSTER_HUB_BYTES.get(dataset)
        return float(exact if exact is not None
                     else 8 * int(np.prod(MONSTER_SHAPES[dataset])))
    if benchmark == "multiverse":
        n_train, n_test, n_channels, m = sizes("multiverse", dataset)
        return 8.0 * (n_train + n_test) * n_channels * m
    return 0.0


def download_root(benchmark, dataset):
    """Where the cell's download lands (None: it downloads nothing)."""
    bench, _ = resolve(benchmark, dataset)
    if bench == "monster":
        return hub_cache()
    if bench == "multiverse":
        return data_home()
    return None


# -------------------------------------------------------------- the disk ---

def _existing(path):
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def free_bytes(path):
    """Free bytes on the volume holding ``path`` -- always the real disk."""
    return float(shutil.disk_usage(str(_existing(path))).free)


def whatif_free():
    """The FREE_ENV what-if in bytes, or None. Previews only."""
    env = os.environ.get(FREE_ENV)
    return float(env) * 1e9 if env else None


def healthy(free):
    """True when ``free`` bytes is a disk in its normal state."""
    return free is not None and free >= HEALTHY_BYTES


def cell_volumes(benchmark, dataset):
    """The paths a cell writes to: the temp dir (streaming caches, joblib
    memmaps) and, if it downloads, where its download lands."""
    paths = [tempfile.gettempdir()]
    root = download_root(benchmark, dataset)
    if root is not None:
        paths.append(root)
    return paths


def cell_free(benchmark, dataset):
    """The least free space, in bytes, over the volumes the cell writes to
    -- what the child reads at each fold's start to know whether a full
    disk that fold meets is the dataset's doing (``healthy``)."""
    return min(free_bytes(p) for p in cell_volumes(benchmark, dataset))


def _allowance(room):
    return min(STREAM_DISK_FRAC * room, room - FLOOR_BYTES)


def _same_volume(a, b):
    try:
        return os.stat(_existing(a)).st_dev == os.stat(_existing(b)).st_dev
    except OSError:
        return True                      # unknown: the stricter reading


def decide(benchmark, arm, dataset, free=None, download=None):
    """The disk verdict for one cell. ``free``: bytes to assume free on
    every volume involved (a what-if or a projection); None reads the disks
    now. ``download``: bytes to assume the cell downloads first (a
    projection that knows an earlier step will have cached the data); None
    asks the cache.

    Read against max(free, HEALTHY_BYTES): EXCLUDED only when the cell
    would not fit even a healthy disk; WAIT when it would fit a healthy
    disk but not this one (module docstring). An exclusion -- of the data
    or of the cache -- outranks a wait: a cell no healthy disk would hold
    is excluded whatever the disk holds now."""
    dl = download_bytes(benchmark, dataset) if download is None else float(download)
    root = download_root(benchmark, dataset)
    f_data = None
    data_wait = None
    if dl > 0:
        f_data = free if free is not None else free_bytes(root)
        ref = max(f_data, HEALTHY_BYTES)
        if dl > ref - FLOOR_BYTES:
            return Verdict(False, f"data download {gb(dl)} GB > {gb(ref)} GB "
                                  f"free - {gb(FLOOR_BYTES)} GB floor"
                                  + ("" if healthy(f_data) else
                                     f" even on a healthy disk ({gb(f_data)} "
                                     f"GB free now)"),
                           download=dl, free=f_data)
        if dl > f_data - FLOOR_BYTES:
            data_wait = Verdict(False, f"waiting: data download {gb(dl)} GB > "
                                       f"{gb(f_data)} GB free - {gb(FLOOR_BYTES)} "
                                       f"GB floor, on a disk below the "
                                       f"{gb(HEALTHY_BYTES)} GB healthy reference",
                                download=dl, free=f_data, wait=True)
    geometry = (_design(arm, sizes(benchmark, dataset)[2])[0]
                if arm in STREAM_GEOMETRY else "per_channel")
    lo, mid, hi = (stream_cache_bytes(benchmark, arm, dataset, r)
                   for r in fill_ratios(geometry))
    ram = STREAM_RAM_MB * 1e6
    if hi <= ram:
        if data_wait is not None:
            return data_wait
        what = (f"download {gb(dl)} GB fits above the floor" if dl
                else "needs no disk")
        if hi:
            what += f"; streaming cache ~{gb(mid)} GB stays in RAM"
        return Verdict(True, what, download=dl, cache=mid, cache_low=lo,
                       cache_high=hi, free=f_data)
    temp = tempfile.gettempdir()
    f_tmp = free if free is not None else free_bytes(temp)
    shared = dl if dl and (free is not None or _same_volume(root, temp)) else 0.0

    def budget(f, allowed):
        room = f - shared
        return (f"min({STREAM_DISK_FRAC} x {gb(room)}, {gb(room)} - "
                f"{gb(FLOOR_BYTES)} floor) = {gb(allowed)} GB allowed ({gb(f)} "
                f"GB free" + (f" - {gb(shared)} GB download)" if shared else ")"))

    def fits(cache, allowed):
        # a cache ferrets keeps in RAM needs no disk at all (a cache
        # RAM-sized at the low ratio must never be an exclusion)
        return cache <= ram or cache <= allowed

    allowance = _allowance(f_tmp - shared)
    ref_free = max(f_tmp, HEALTHY_BYTES)
    ref_allowance = _allowance(ref_free - shared)
    sizes_txt = (f"streaming cache ~{gb(mid)} GB (ferrets's estimate, "
                 f"predicted; {gb(lo)}-{gb(hi)} GB over the measured fill ratios)")
    common = dict(download=dl, cache=mid, cache_low=lo, cache_high=hi,
                  allowance=allowance, free=f_tmp)
    if not fits(lo, ref_allowance):
        return Verdict(False, f"{sizes_txt} > {budget(ref_free, ref_allowance)}"
                              + ("" if healthy(f_tmp) else
                                 f" on a healthy disk, {gb(f_tmp)} GB free now")
                              + " even at the lowest measured fill", **common)
    if data_wait is not None:
        return data_wait
    if fits(hi, allowance):
        return Verdict(True, f"{sizes_txt} <= {budget(f_tmp, allowance)}",
                       **common)
    if healthy(f_tmp):              # then lo fits: ref_allowance == allowance
        return Verdict(True, f"borderline: {sizes_txt} vs "
                             f"{budget(f_tmp, allowance)}; ferrets's own "
                             f"probe decides at fit time", borderline=True,
                       **common)
    return Verdict(False, f"waiting: {sizes_txt} vs {budget(f_tmp, allowance)}, "
                          f"on a disk below the {gb(HEALTHY_BYTES)} GB healthy "
                          f"reference (a healthy disk would hold it)",
                   wait=True, **common)


def error_text(reason):
    """The banked error of an excluded fold."""
    return f"{DISK_PREFIX}: {reason}"


def skip_rows(cell, verdict, n_threads):
    """One banked row per fold of an excluded cell."""
    return [new_row(cell.benchmark, cell.dataset, cell.arm, f,
                    split=split_kind(cell.benchmark, cell.dataset, f),
                    n_threads=int(n_threads), error=error_text(verdict.reason))
            for f in cell.folds]


# --------------------------------------------- previews: the step's disk ---

def current_caches():
    """{(benchmark, dataset): bytes on disk} of the downloaded caches the
    reaper manages (benchkit/reap.py): Multiverse .npz, MONSTER hub repos."""
    held = {}
    for bench in ("multiverse", "monster"):
        for name in datasets(bench):
            n = size_of(cache_paths(bench, name))
            if n:
                held[(bench, name)] = n
    return held


@dataclass(frozen=True)
class Projection:
    free: float                 # bytes free when the step starts
    now: float                  # bytes free now
    reclaimed: float            # caches the reaper will have dropped by then
    downloaded: float           # data earlier steps download and leave for it
    fetched: frozenset          # (benchmark, dataset) cached by then, not now


def project(plan, i, now, extra=(), held=None, runnable=None):
    """The free disk when run step ``plan[i]`` starts, if every step before
    it completes: ``now`` + the caches no unbanked cell of step i or later
    reads (the reaper drops them as the earlier steps finish -- bench.py
    _reap_after -- and the queue drops those NO step reads when it starts,
    bench.py _reap_unneeded_caches) - the data an earlier step downloads
    that step i or later still reads (a MONSTER dataset an earlier cell
    fetched, or a Multiverse dataset a ``prefetch`` step before i names:
    its ``datasets``, None = every Multiverse problem). ``plan``: [(step,
    its unbanked cells, or None for a non-run step)], in queue order;
    ``extra``: cells of steps outside the plan that also keep caches (the
    other target); ``runnable``: a predicate on a cell (scope.
    runnable_by_gate) -- a dataset counts as read only through cells it
    accepts, so one whose every cell the gate refuses is neither kept nor
    fetched (None: every cell counts). A preview only: the run reads the
    real disk before each cell, and within a step the free disk moves as
    datasets are fetched and reaped."""
    held = current_caches() if held is None else held
    ok = runnable if runnable is not None else (lambda cell: True)
    later, earlier = set(), set()
    prefetched_all, prefetched = False, set()
    for k, (step, cells) in enumerate(plan):
        if k < i and step.kind == "prefetch" and step.benchmark == "multiverse":
            if step.datasets is None:
                prefetched_all = True
            else:
                prefetched.update(step.datasets)
        if step.kind != "run" or not cells:
            continue
        (later if k >= i else earlier).update(
            resolve(c.benchmark, c.dataset) for c in cells if ok(c))
    later.update(resolve(c.benchmark, c.dataset) for c in extra if ok(c))
    reclaimed = float(sum(n for pair, n in held.items() if pair not in later))
    fetched = frozenset(
        pair for pair in later
        if pair[0] in ("multiverse", "monster") and pair not in held
        and (pair in earlier
             or (pair[0] == "multiverse"
                 and (prefetched_all or pair[1] in prefetched))))
    downloaded = float(sum(resident_bytes(*pair) for pair in fetched))
    return Projection(now + reclaimed - downloaded, now, reclaimed,
                      downloaded, fetched)


def decide_projected(cell, projection):
    """decide() at a projected step start (bench.py queue --dry-run)."""
    pair = resolve(cell.benchmark, cell.dataset)
    return decide(cell.benchmark, cell.arm, cell.dataset, free=projection.free,
                  download=0.0 if pair in projection.fetched else None)
