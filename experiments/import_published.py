"""Import the published UCR reference results: results/published_ucr.csv.

Source: the reference archive behind timeseriesclassification.com (the one
aeon's ``benchmarking.results_loaders`` reads): 112 UCR datasets x 30
resamples per estimator, quality measures only. It publishes no fit time,
no predict time and no memory, so these rows fill accuracy columns and
never a cost column.

Resample 0 is the published train/test split, the split our fold 0 runs;
resamples 1-29 are the archive's seeded resamples. Rows carry
``split="published"`` for fold 0 and ``split="resample"`` for folds 1-29.
Three measures are read: accuracy, balanced accuracy and F1. The archive's
F1 is the weighted average: on folds both hold, it tracks our own
``f1_weighted`` of QUANT, Hydra and MiniRocket, not ``f1_macro``.

Rows are in the bench schema (experiments/benchkit/schema.py), cost columns
empty. Model names carry a ``_pub`` suffix so a concatenation with measured
rows can never confuse the two.

    python experiments/import_published.py
    python experiments/import_published.py --estimators FreshPRINCE --merge
                                          # add or refresh one estimator, the
                                          # other estimators' rows untouched
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchkit.data import datasets  # noqa: E402
from experiments.benchkit.schema import COLUMNS, new_row  # noqa: E402

BASE = ("https://timeseriesclassification.com/results/ReferenceResults/"
        "classification/{measure}/{est}_{measure}.csv")
SOURCE = "published:timeseriesclassification.com"
FOLDS = tuple(range(30))   # 0 = published split; 1-29 = seeded resamples
F1_COLUMN = "f1_weighted"
# The archive serves MultiRocket-Hydra under the name "MR-Hydra".
DEFAULT = ("LiteTime", "InceptionTime", "ResNet", "CNN",
           "QUANT", "Hydra", "MiniROCKET", "MrSQM", "WEASEL-2.0",
           "TDE", "cBOSS",
           "HC2", "MR-Hydra", "RDST", "DrCIF", "RIST", "ROCKET",
           "Arsenal", "FreshPRINCE")


def fetch(est, measure):
    """dataset x resample table of one (estimator, measure)."""
    frame = pd.read_csv(BASE.format(est=est, measure=measure), index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = [int(c) for c in frame.columns]
    return frame


def build_rows(estimators, wanted, f1_column=F1_COLUMN, fetch=fetch):
    """Bench rows of the estimators on the wanted datasets, folds 0-29."""
    rows = []
    for est in estimators:
        measures = {}
        for name, column in (("accuracy", "acc"), ("balacc", "balanced_acc"),
                             ("f1", f1_column)):
            try:
                measures[column] = fetch(est, name)
            except Exception as error:                       # noqa: BLE001
                print(f"  {est}: no {name} ({type(error).__name__})")
        if "acc" not in measures:
            continue
        covered = [d for d in wanted if d in measures["acc"].index]
        for dataset in covered:
            for fold in FOLDS:
                values = {c: float(frame.loc[dataset, fold])
                          for c, frame in measures.items()
                          if fold in frame.columns}
                if np.isnan(values.get("acc", np.nan)):
                    continue
                rows.append(new_row("ucr", dataset, f"{est}_pub", fold,
                                    split="published" if fold == 0 else "resample",
                                    os=SOURCE, **values))
        print(f"  {est:15s} {len(covered):3d}/{len(wanted)} datasets"
              f" x {len(FOLDS)} folds")
    return rows


def write_atomic(frame, out, attempts=20, pause=0.5):
    """Write ``frame`` to ``out`` through a temp file in the same directory
    and ``os.replace``: a reader sees the old file or the new one, never a
    half-written one. Windows refuses the replace while another process
    holds the file open, so it is retried for about ten seconds."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".",
                                    suffix=".tmp")
    os.close(handle)
    tmp = Path(name)
    try:
        frame.to_csv(tmp, index=False)
        for attempt in range(attempts):
            try:
                os.replace(tmp, out)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(pause)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimators", default=",".join(DEFAULT),
                        help="comma-separated published estimator names")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" /
                                             "published_ucr.csv"))
    parser.add_argument("--merge", action="store_true",
                        help="keep the other estimators' rows of --out and "
                             "replace only those of --estimators")
    args = parser.parse_args(argv)

    wanted = list(datasets("ucr"))
    print(f"published reference archive -> {len(wanted)} UCR datasets, "
          f"folds 0-29 (fold 0 is the published split)")
    rows = build_rows([e.strip() for e in args.estimators.split(",") if e.strip()],
                      wanted)
    out = Path(args.out)
    # a network failure or a bad estimator list must never replace a good
    # CSV with an empty or much smaller one
    old = pd.read_csv(out) if out.exists() else None
    if args.merge and old is not None:
        new = {r["model"] for r in rows}
        rows = old[~old["model"].isin(new)].to_dict("records") + rows if rows else []
    existing = 0 if old is None else len(old)
    if not rows or (existing and existing > 2 * len(rows)):
        print(f"refusing to write {out}: {len(rows)} new row(s) vs {existing} "
              f"already there (every fetch failed, or the archive shrank by "
              f"more than half); leaving the file alone")
        raise SystemExit(2)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    write_atomic(frame, out)
    print(f"\n{len(frame)} rows -> {out}")
    return frame


if __name__ == "__main__":
    main()
