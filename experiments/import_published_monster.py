"""Import the MONSTER results workbook: results/published_monster.csv.

Source: ``monster_final_results.xlsx`` of the MONSTER repository
(Navidfoumani/monster on GitHub). The file the released rows were read from
has the SHA256 checksum in WORKBOOK_SHA256, the one the paper prints; a
workbook with another checksum is refused. When the workbook path does not
exist, the file is downloaded there from the repository first (its current
revision, which must still have that checksum).

Every worksheet is exported as long-form rows. The sheets overlap and
sometimes differ, so ``source_sheet`` is kept and no sheet is preferred or
averaged. Dataset names that differ from the hub's only by case are
unified; ``source_dataset`` keeps the workbook's spelling. ``accuracy`` is
1 - ``classification_error``. Reading the workbook needs openpyxl.

    python experiments/import_published_monster.py [WORKBOOK] [--out CSV]
"""

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "results" / "published_monster.xlsx"
CSV = REPO_ROOT / "results" / "published_monster.csv"
URL = ("https://raw.githubusercontent.com/Navidfoumani/monster/main/"
       "monster_final_results.xlsx")
SOURCE = "Navidfoumani/monster:main/monster_final_results.xlsx"
WORKBOOK_SHA256 = "a231ce5b316bd693c00df78b8f45d493bccdce1769bda49e34065fda76fb63cd"


def canonical_dataset(name):
    """Unify capitalization-only aliases while retaining the source name."""
    aliases = {
        "crowdsourced": "CrowdSourced",
        "dreamera": "DREAMERA",
        "dreamerv": "DREAMERV",
    }
    return aliases.get(str(name).casefold(), str(name))


def check_digest(path):
    """Refuse a workbook whose SHA256 is not the released one."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != WORKBOOK_SHA256:
        raise SystemExit(f"{path}: SHA256 {digest}, expected {WORKBOOK_SHA256} "
                         f"(the workbook the released rows were read from)")


def long_rows(sheets):
    """One long-form frame from {sheet name: frame}."""
    frames = []
    for sheet, frame in sheets.items():
        frame = frame.copy()
        frame.insert(0, "source_sheet", sheet)
        frame["source_dataset"] = frame["dataset"].astype(str)
        frame["dataset"] = frame["source_dataset"].map(canonical_dataset)
        frame["accuracy"] = 1.0 - frame["classification_error"]
        frame["source"] = SOURCE
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workbook", nargs="?", default=str(WORKBOOK),
                        help="the workbook (downloaded there when absent)")
    parser.add_argument("--out", default=str(CSV))
    args = parser.parse_args(argv)
    workbook, out = Path(args.workbook), Path(args.out)
    if not workbook.exists():
        workbook.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            workbook.write_bytes(response.read())
    check_digest(workbook)

    result = long_rows(pd.read_excel(workbook, sheet_name=None))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Workbook: {workbook}")
    print(f"Long-form rows: {len(result)} -> {out}")
    print(result.groupby(["source_sheet", "method"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
