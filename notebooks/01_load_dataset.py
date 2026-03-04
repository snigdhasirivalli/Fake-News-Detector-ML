"""
01_load_dataset.py
------------------
Quick sanity-check: load the WELFake dataset and preview the first 5 rows.

Prerequisites:
  1. Download WELFake_Dataset.csv from Kaggle:
     https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
  2. Place the CSV in:  ../data/WELFake_Dataset.csv

Dataset columns:
  - Unnamed: 0  : original row index
  - title       : article headline
  - text        : full article body
  - label       : 0 = Real News, 1 = Fake News
"""

import pathlib
import pandas as pd

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "WELFake_Dataset.csv"


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Load the WELFake CSV and return a cleaned DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌  Dataset not found at: {path}\n"
            "    Please download it from:\n"
            "    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            "    and place it in the data/ folder."
        )

    df = pd.read_csv(path)
    # Drop the unnamed index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def main():
    print("=" * 60)
    print("  📰  WELFake Dataset — Quick Preview")
    print("=" * 60)

    df = load_dataset(DATA_PATH)

    print(f"\n  Total rows    : {len(df):,}")
    print(f"  Total columns : {len(df.columns)}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"\n  Label distribution:")
    label_map = {0: "Real", 1: "Fake"}
    counts = df["label"].value_counts().rename(index=label_map)
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"    {label:<6} → {count:>6,} articles  ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("  First 5 rows:")
    print("=" * 60)
    pd.set_option("display.max_colwidth", 60)
    print(df.head())
    print("=" * 60)


if __name__ == "__main__":
    main()
