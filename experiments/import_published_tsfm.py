"""Import the published UCR accuracies of time-series foundation models:
results/published_tsfm.csv.

Source: Feofanov et al., "MantisV2: Closing the Zero-Shot Gap in Time Series
Classification with Synthetic Data and Test-Time Strategies",
arXiv:2602.17868v1, the appendix tables of its zero-shot comparison with the
logistic-regression head on UCR (tab:sota-ucr-res-logreg-a and -b, all 128
datasets), read from the LaTeX source of the e-print. The e-print's SHA256
is checked against EPRINT_SHA256.

Models: the latest version of each, MantisV2, MOMENT (MOMENT-1-large), TiRex,
Chronos-2 and NuTime. Datasets: the benchmark's 112 UCR datasets; every one
must be in the source.

The source's protocol: frozen encoder, each series resized to length 512,
the embedding of one intermediate layer per model, then StandardScaler and
scikit-learn's LogisticRegression on the default train/test split; one value
per cell. The rows are therefore fold 0 with accuracy only: no balanced
accuracy, F1, time or memory. Every imported column's mean over the
source's datasets must equal the table's Average row.

Rows are in the bench schema (experiments/benchkit/schema.py): model
"<name>_pub", split "published".

    python experiments/import_published_tsfm.py [--tex main.tex] [--out CSV]
"""

import argparse
import hashlib
import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchkit.data import datasets  # noqa: E402
from experiments.benchkit.schema import COLUMNS, new_row  # noqa: E402
from experiments.import_published import write_atomic  # noqa: E402

ARXIV_ID = "2602.17868v1"
URL = f"https://arxiv.org/e-print/{ARXIV_ID}"
EPRINT_SHA256 = "6bbf81a7acc78b522096d8695a4e545ddee90127cf2845d4672e00fd1e3a3453"
SOURCE = f"published:arXiv:{ARXIV_ID} (zero-shot, logistic regression)"
TABLES = ("tab:sota-ucr-res-logreg-a", "tab:sota-ucr-res-logreg-b")
# paper column -> model name (the "_pub" suffix is added on the rows)
MODELS = {"MantisV2": "MantisV2", "MOMENT": "MOMENT", "TiRex": "TiRex",
          "Chronos2": "Chronos-2", "NuTime": "NuTime"}
SUMMARY_ROWS = ("Average", "Best Counts")


def fetch_source(url=URL):
    """main.tex of the pinned e-print, its SHA256 checked."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EPRINT_SHA256:
        raise SystemExit(f"{url}: SHA256 {digest}, expected {EPRINT_SHA256}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        return tar.extractfile("main.tex").read().decode("utf-8")


def _plain(cell):
    return re.sub(r"\\[a-z]+\{|[{}$]", "", cell).strip()


def cell_value(cell):
    """The accuracy in a cell (``0.9``, ``\\textbf{0.86}``,
    ``$0.82_{\\pm 0.026}$``), None for ``\\texttt{NaN}``."""
    if "nan" in cell.lower():
        return None
    number = re.search(r"\d+(?:\.\d+)?", cell)
    if number is None:
        raise ValueError(f"no number in cell {cell!r}")
    return float(number.group())


def parse_table(tex, label):
    """(columns, {row name: cells}) of the tabular after ``\\label{label}``."""
    start = tex.index(r"\label{" + label + "}")
    body = tex[start:tex.index(r"\end{tabular}", start)]
    lines = [line.strip() for line in body.splitlines()]
    lines = [line for line in lines if line and not line.startswith("%")]
    header = next(line for line in lines if line.startswith("&"))
    columns = [_plain(c) for c in header.rstrip("\\").split("&")[1:]]
    rows = {}
    for line in lines:
        if line.startswith("&") or not line.endswith("\\\\"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        if len(cells) != len(columns) + 1:
            raise ValueError(f"{label}: {len(cells)} cells in {line[:60]!r}")
        rows[_plain(cells[0])] = cells[1:]
    return columns, rows


def read_tables(tex, labels=TABLES):
    """dataset x paper column accuracies over the tables, checked against
    the Average row (the last table carries it)."""
    frames, average = [], None
    for label in labels:
        columns, rows = parse_table(tex, label)
        if "Average" in rows:
            average = dict(zip(columns, rows["Average"]))
        frames.append(pd.DataFrame(
            {name: [cell_value(c) for c in cells]
             for name, cells in rows.items() if name not in SUMMARY_ROWS},
            index=columns).T)
    table = pd.concat(frames)
    if table.index.duplicated().any():
        raise ValueError(f"{labels}: duplicated datasets "
                         f"{sorted(table.index[table.index.duplicated()])}")
    if average is None:
        raise ValueError(f"{labels}: no Average row")
    for column in MODELS:
        printed = cell_value(average[column])
        if abs(table[column].mean() - printed) > 5e-5:
            raise ValueError(f"{labels} {column}: mean {table[column].mean():.5f}"
                             f" against the printed Average {printed}")
    return table


def build_rows(tex, ucr_names):
    """(rows for the MODELS on the given UCR datasets, the source's datasets
    left out). A wanted dataset absent from the source is an error, as is a
    missing value of an imported model."""
    table = read_tables(tex)
    wanted = set(ucr_names)
    if wanted - set(table.index):
        raise ValueError(f"UCR datasets missing from the source: "
                         f"{sorted(wanted - set(table.index))}")
    rows = []
    for dataset in sorted(wanted):
        for column, model in MODELS.items():
            acc = table.at[dataset, column]
            if pd.isna(acc):
                raise ValueError(f"ucr {dataset} {column}: NaN")
            rows.append(new_row("ucr", dataset, f"{model}_pub", 0,
                                split="published", acc=float(acc), os=SOURCE))
    return rows, sorted(set(table.index) - wanted)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", help="a local copy of the source's main.tex "
                                      "(default: fetch the e-print)")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" /
                                             "published_tsfm.csv"))
    args = parser.parse_args(argv)
    tex = (Path(args.tex).read_text(encoding="utf-8") if args.tex
           else fetch_source())
    rows, left_out = build_rows(tex, datasets("ucr"))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    write_atomic(frame, args.out)
    means = frame.groupby("model").acc.mean().round(4).to_dict()
    print(f"ucr: {frame.dataset.nunique()} datasets, mean acc {means}")
    print(f"  left out: {left_out}")
    print(f"{len(frame)} rows -> {args.out}  (accuracy only, fold 0)")
    return frame


if __name__ == "__main__":
    main()
