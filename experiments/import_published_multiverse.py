"""Import the Multiverse archive's published results:
results/published_multiverse.csv.

Source: the aeon-toolkit/multiverse repository on GitHub, at the commit
pinned in REF (the one the released file was read from). It holds one table
per estimator and metric (``results/multiverse/<est>/<est>_<metric>.csv``:
a ``Resamples:,0`` header, then one ``dataset,value`` line per dataset), for
resample 0 only, the archive's default train/test split, which is the split
our fold 0 runs. It publishes no memory and no comparable runtime, so these
rows fill accuracy columns and never a cost column. Three metrics are read:
accuracy, balanced accuracy and F1. The archive's F1 is the macro average:
it tracks our own ``f1_macro`` of QUANT, not ``f1_weighted``.

Rows are in the bench schema (experiments/benchkit/schema.py): fold 0,
split "published", model "<est>_pub", cost columns empty. Every dataset the
archive holds is imported. The commit and the estimators read are recorded
in results/published_multiverse.source.json.

    python experiments/import_published_multiverse.py [--ref SHA] [--out CSV]
"""

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchkit.schema import COLUMNS, new_row  # noqa: E402
from experiments.import_published import write_atomic  # noqa: E402

REF = "5fdf986eaed93369fe8563c58e183bfbb3221559"
RAW = ("https://raw.githubusercontent.com/aeon-toolkit/multiverse/{ref}/"
       "results/multiverse/{est}/{est}_{metric}.csv")
TREE = "https://api.github.com/repos/aeon-toolkit/multiverse/git/trees/{ref}?recursive=1"
SOURCE = "published:aeon-toolkit/multiverse"
F1_COLUMN = "f1_macro"
METRICS = (("accuracy", "acc"), ("balacc", "balanced_acc"), ("f1", None))


def list_estimators(ref=REF):
    """(estimator names, commit sha) from the repository tree at ref."""
    with urllib.request.urlopen(TREE.format(ref=ref), timeout=30) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise SystemExit("GitHub truncated the tree response; the estimator "
                         "list would be short")
    names = sorted({p["path"].split("/")[2] for p in tree["tree"]
                    if p["path"].startswith("results/multiverse/")
                    and p["path"].endswith("_accuracy.csv")})
    return names, tree.get("sha")


def fetch_table(est, metric, ref=REF, url=None):
    """dataset -> value for one (estimator, metric) table (``url``: a local
    or remote table in place of the archive's)."""
    raw = pd.read_csv(url or RAW.format(ref=ref, est=est, metric=metric),
                      header=None)
    head = [str(v) for v in raw.iloc[0].tolist()]
    if not head[0].startswith("Resamples") or len(head) != 2:
        raise ValueError(f"{est}_{metric}: expected one resample column, "
                         f"got header {head}")
    body = raw.iloc[1:]
    return pd.Series(body[1].astype(float).to_numpy(),
                     index=body[0].astype(str).to_numpy(), name=metric)


def build_rows(estimators, f1_column=F1_COLUMN, fetch=fetch_table):
    """Bench rows of the estimators, fold 0, every dataset of their tables."""
    rows = []
    for est in estimators:
        tables = {}
        for metric, column in METRICS:
            column = f1_column if metric == "f1" else column
            try:
                tables[column] = fetch(est, metric)
            except Exception as error:                        # noqa: BLE001
                print(f"  {est}: no {metric} ({type(error).__name__})")
        if "acc" not in tables:
            continue
        for dataset, acc in tables["acc"].items():
            if np.isnan(acc):
                continue
            values = {c: float(s[dataset]) for c, s in tables.items()
                      if dataset in s.index and not np.isnan(s[dataset])}
            rows.append(new_row("multiverse", dataset, f"{est}_pub", 0,
                                split="published", os=SOURCE, **values))
        print(f"  {est:18s} {len(tables['acc']):3d} datasets")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=REF,
                        help="the archive's commit or branch (default: the "
                             "commit the released file was read from)")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" /
                                             "published_multiverse.csv"))
    args = parser.parse_args(argv)
    names, sha = list_estimators(args.ref)
    print(f"Multiverse archive @ {args.ref} ({sha}): {len(names)} estimators, "
          f"resample 0 = the published split")
    rows = build_rows(names, fetch=lambda e, m: fetch_table(e, m, args.ref))
    out = Path(args.out)
    # a fetch that failed entirely must never replace a good CSV with an
    # empty or much smaller one
    existing = len(pd.read_csv(out)) if out.exists() else 0
    if not rows or (existing and existing > 2 * len(rows)):
        print(f"refusing to write {out}: {len(rows)} new row(s) vs {existing} "
              f"already there; leaving the file alone")
        raise SystemExit(2)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    write_atomic(frame, out)
    out.with_suffix(".source.json").write_text(json.dumps(dict(
        url=RAW, ref=args.ref, sha=sha, estimators=names, f1_column=F1_COLUMN,
        fetched=datetime.datetime.now().isoformat(timespec="seconds")),
        indent=1), encoding="utf-8")
    print(f"\n{len(frame)} rows -> {out}")
    return frame


if __name__ == "__main__":
    main()
