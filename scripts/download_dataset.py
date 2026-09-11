"""Download a real, comprehensive retail/e-commerce dataset for testing the Streamlit uploader.

Fetches the classic UCI "Online Retail" transaction dataset (invoice-level UK online retail
orders, Dec 2010-Dec 2011: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country) from a public raw-CSV mirror -- no API key or auth required, unlike the
Kaggle API. Saves it to `data/real_dataset.csv`, ready to drag into the Streamlit uploader.

Usage:
    uv run python scripts/download_dataset.py
"""

import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Tried in order; the first that downloads and parses as a real CSV wins. Both are stable,
#: long-lived mirrors of the same underlying UCI "Online Retail" dataset (541k+ rows) -- the
#: official UCI source only serves .xlsx, which is why a CSV mirror is used instead.
CANDIDATE_URLS = [
    # Databricks' own "Spark: The Definitive Guide" companion repo -- an official, widely
    # depended-on mirror, unlikely to move or disappear.
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv",
    # Independent cleaned-CSV mirror of the same source dataset, as a fallback.
    "https://raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version/master/Cleaned_UCI_Online_Sale_Dataset.csv",
]

MIN_ROWS = 50_000
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_PATH = _REPO_ROOT / "data" / "real_dataset.csv"
_CHUNK_SIZE = 1024 * 1024
_TIMEOUT_SECONDS = 60


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, overwriting any existing file."""
    request = urllib.request.Request(url, headers={"User-Agent": "adia-dataset-downloader"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
        total = response.length or 0
        written = 0
        with open(dest, "wb") as f:
            while chunk := response.read(_CHUNK_SIZE):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    mb_done, mb_total = written / 1e6, total / 1e6
                    msg = f"\r  downloaded {mb_done:6.1f} MB / {mb_total:6.1f} MB ({pct:5.1f}%)"
                    print(msg, end="")
                else:
                    print(f"\r  downloaded {written / 1e6:6.1f} MB", end="")
    print()


def count_rows(path: Path) -> int:
    """Count data rows (excluding the header) in a CSV file."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def main() -> int:
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(CANDIDATE_URLS, start=1):
        print(f"[{i}/{len(CANDIDATE_URLS)}] downloading real dataset from:\n  {url}")
        try:
            download(url, DEST_PATH)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        try:
            rows = count_rows(DEST_PATH)
        except (OSError, csv.Error) as exc:
            print(f"  downloaded file is not a readable CSV: {exc}", file=sys.stderr)
            DEST_PATH.unlink(missing_ok=True)
            continue

        if rows < MIN_ROWS:
            print(f"  only {rows:,} rows (< {MIN_ROWS:,} required) -- trying next source")
            DEST_PATH.unlink(missing_ok=True)
            continue

        size_mb = DEST_PATH.stat().st_size / 1e6
        print(f"OK: saved {rows:,} rows ({size_mb:.1f} MB) to {DEST_PATH.relative_to(_REPO_ROOT)}")
        return 0

    print("All candidate sources failed or were too small.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
