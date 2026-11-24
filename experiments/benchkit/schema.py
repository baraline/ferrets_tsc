"""The one row layout every benchmark CSV shares."""

import platform
import sys

COLUMNS = [
    "benchmark", "dataset", "model", "fold", "split",
    "n_train", "n_test", "n_channels", "m", "n_classes",
    "acc", "f1_macro", "f1_weighted", "balanced_acc",
    "time_fit", "time_pred",
    "mem_base_mb", "mem_peak_mb", "mem_median_mb", "mem_peak_rss_mb",
    "mem_wset_peak_mb", "mem_samples",
    "model_bytes", "n_features", "n_threads", "device",
    "os", "python", "sklearn", "aeon", "started_at", "error",
]
KEY = ("dataset", "model", "fold")
# which data the row was measured on (tsml-eval convention): fold 0 of a
# real dataset is the archive's published split untouched, fold >= 1 a
# stratified resample seeded by the id, and a cost-grid cell is generated.
# Descriptive: the key does not include it.
SPLITS = ("published", "resample", "synthetic")
# error prefix of a row the feasibility gate declined to run: a PREDICTION,
# not a measurement (benchkit/gate.py). Delete the row to force the cell.
SKIP_PREFIX = "not run: "
# error prefix of a cell EXCLUDED because running it would exhaust the disk
# (benchkit/disk.py): one of the SKIP_PREFIX family -- never a measured
# failure -- and stated by the paper as excluded rather than out of reach.
DISK_PREFIX = SKIP_PREFIX + "disk"


def new_row(benchmark, dataset, model, fold, **fields):
    """A row with every column present (None where unmeasured)."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise KeyError(f"not schema columns: {sorted(unknown)}")
    row = {c: None for c in COLUMNS}
    row.update(benchmark=benchmark, dataset=dataset, model=model,
               fold=int(fold))
    row.update(fields)
    return row


def os_name():
    """'win32', 'wsl2-linux' (the MrSQM child) or sys.platform."""
    if sys.platform.startswith("linux") and \
            "microsoft" in platform.release().lower():
        return "wsl2-linux"
    return sys.platform


def env_versions():
    import aeon
    import sklearn
    return dict(os=os_name(), python=platform.python_version(),
                sklearn=sklearn.__version__, aeon=aeon.__version__)
