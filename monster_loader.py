import logging
import os

import numpy as np
from aeon.utils.numba.general import z_normalise_series_3d
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

univariate_monster_datasets = [
    "CornellWhaleChallenge",
    "AudioMNIST",
    "WhaleSounds",
    "Pedestrian",
    "FruitFlies",
    "AudioMNIST-DS",
    "Traffic",
    "LakeIce",
    "MosquitoSound",
    "InsectSound",
]


def _fetch(dataset_name, fold):
    """(path of <name>_X.npy, y, test_index) of one MONSTER dataset and
    fold: the hub's X file as downloaded (#Samples, #Channel, #Length), the
    labels, and the fold's test rows."""
    repo_id = f"monster-monash/{dataset_name}"

    # Download data
    data_path = hf_hub_download(
        repo_id=repo_id, filename=f"{dataset_name}_X.npy", repo_type="dataset"
    )
    # Download labels
    label_filename = f"{dataset_name}_Y.npy"
    try:
        label_path = hf_hub_download(
            repo_id=repo_id, filename=label_filename, repo_type="dataset"
        )
    except:
        label_filename = f"{dataset_name}_y.npy"
        label_path = hf_hub_download(
            repo_id=repo_id, filename=label_filename, repo_type="dataset"
        )
    y = np.load(label_path)
    # Load test indices
    try:
        test_index_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"test_indices_fold_{fold}.txt",
            repo_type="dataset",
        )
        test_index = np.loadtxt(test_index_path, dtype=int)
    except Exception as e:
        logger.error(f"Failed to load test indices: {e}")
        raise
    return data_path, y, test_index


def load_monster(dataset_name, fold):
    """The LEGACY loader: every series z-normalised per channel over the whole
    dataset before the split, as the first MONSTER runs and the benchmark up
    to 2026-10-08 read it (the cost benchmark's MONSTER anchor rows measured
    before that date among them). The benchmark itself reads
    ``load_monster_raw`` since 2026-10-08; this function is kept
    bit-identical so those rows stay reproducible."""
    data_path, y, test_index = _fetch(dataset_name, fold)
    X = np.load(data_path, mmap_mode="r")  # (#Samples, #Channel, #Length)
    X = z_normalise_series_3d(X)

    test_bool_index = np.zeros(len(y), dtype=bool)
    test_bool_index[test_index] = True
    return (
        X[~test_bool_index],
        y[~test_bool_index],
        X[test_bool_index],
        y[test_bool_index],
    )


def _npy_header(f):
    """(shape, dtype, data offset) of an open .npy file, C order only. Read
    with numpy's public header readers, so that no memory map is made."""
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"unsupported .npy format version {version}")
    if fortran or dtype.hasobject or len(shape) < 2:
        raise ValueError(f"expected a C-ordered numeric (n, ...) array, got "
                         f"shape {shape}, dtype {dtype}, fortran_order={fortran}")
    return shape, dtype, f.tell()


def _split_rows(path, is_test, dtype, chunk_mb):
    """(X_train, X_test): the rows of the .npy file at ``path`` that
    ``is_test`` leaves out and keeps, in file order, as C-contiguous
    ``dtype`` arrays.

    The file is read once, front to back, in chunks of about ``chunk_mb`` MB
    of shipped values with plain reads: no memory map, so the pages of an
    8 GB float64 file do not pile up in the process's working set beside the
    float64 result (measured 2026-10-08: with a map the load peaked at
    1 + itemsize/8 times the result, 2.1x for float64 files, 1.55x float32,
    1.27x int16). The peak is the two result arrays plus one chunk."""
    with open(path, "rb") as f:
        shape, shipped, offset = _npy_header(f)
        if len(is_test) != shape[0]:
            raise ValueError(f"{path}: {shape[0]} rows but {len(is_test)} labels")
        inner = shape[1:]
        row_values = int(np.prod(inner, dtype=np.int64))
        have = os.fstat(f.fileno()).st_size
        need = offset + shape[0] * row_values * shipped.itemsize
        if have < need:
            raise ValueError(f"{path}: truncated, {have} bytes of {need}")
        n_test = int(is_test.sum())
        X_train = np.empty((shape[0] - n_test,) + inner, dtype=dtype)
        X_test = np.empty((n_test,) + inner, dtype=dtype)
        step = max(1, int(chunk_mb * 2 ** 20)
                   // max(1, row_values * shipped.itemsize))
        j_train = j_test = 0
        for a in range(0, shape[0], step):
            b = min(shape[0], a + step)
            rows = np.fromfile(f, dtype=shipped, count=(b - a) * row_values)
            rows = rows.reshape((b - a,) + inner)
            t = is_test[a:b]
            k = int(t.sum())
            X_test[j_test:j_test + k] = rows[t]
            X_train[j_train:j_train + (b - a - k)] = rows[~t]
            j_test += k
            j_train += b - a - k
    return X_train, X_test


def load_monster_raw(dataset_name, fold, dtype=np.float64, chunk_mb=64):
    """(X_train, y_train, X_test, y_test) of one MONSTER dataset and fold, X
    exactly as the hub ships it, only cast to ``dtype`` (default float64):
    no normalisation of any kind.

    This is what the MONSTER archive's own evaluation feeds its baselines
    (2026-10-08, Navidfoumani/monster: ``src/data_loader.py`` loads the .npy
    memory-mapped and ``dataset_class.__getitem__`` does ``x.astype(
    np.float32)`` and nothing else; Hydra and Quant read mmap batches and
    ``torch.tensor(X.astype(np.float32))``; the paper's extremely randomised
    trees "train directly on the 'raw' time series data"). Any scaling is the
    method's own business. Rows keep the hub's order within each side, so
    this is ``load_monster`` minus its z-normalisation, row for row.

    float64 rather than the shipped float32 because every other benchmark's
    loader returns float64, the JIT warm-up collection is float64 (a float32
    first call would compile new numba specialisations inside the fit timer)
    and the feasibility gate prices 8 bytes per value. The peak memory is the
    float64 result plus one ``chunk_mb`` chunk (``_split_rows``)."""
    data_path, y, test_index = _fetch(dataset_name, fold)
    is_test = np.zeros(len(y), dtype=bool)
    is_test[np.atleast_1d(test_index)] = True
    X_train, X_test = _split_rows(data_path, is_test, dtype, chunk_mb)
    return X_train, y[~is_test], X_test, y[is_test]
