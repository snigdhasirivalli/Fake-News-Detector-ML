"""
src/preprocess.py
-----------------
Text preprocessing pipeline for the Fake News Detection project.

Pipeline steps:
  1. Lowercase conversion
  2. Remove special characters & numbers (Regex)
  3. Tokenize into words
  4. Remove NLTK English stopwords
  5. Porter Stemming (reduce words to their root form)

Usage (standalone):
    python src/preprocess.py

Usage (as a module):
    from src.preprocess import clean_text, load_data
"""

import re
import pathlib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are available
# ---------------------------------------------------------------------------
def _download_nltk_resources():
    """Download NLTK data silently if not already present."""
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [NLTK] Downloading '{pkg}'...")
            nltk.download(pkg, quiet=True)

_download_nltk_resources()

# ---------------------------------------------------------------------------
# Global objects (created once for performance)
# ---------------------------------------------------------------------------
_STOP_WORDS = set(stopwords.words("english"))
_STEMMER    = PorterStemmer()

# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean a single news article string through the full NLP pipeline.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Cleaned, stemmed, stopword-free text.
    """
    if not isinstance(text, str):
        return ""

    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 3 — Remove special characters, punctuation, and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 — Collapse multiple whitespace into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5 — Tokenize
    tokens = word_tokenize(text)

    # Step 6 — Remove stopwords and very short tokens (len < 2)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # Step 7 — Porter Stemming
    tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "WELFake_Dataset.csv"

def load_data(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the WELFake CSV, drop rows with missing 'text', and return a clean
    DataFrame with columns: [title, text, label].
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌  Dataset not found: {path}\n"
            "    Download WELFake_Dataset.csv from Kaggle and place it in data/"
        )

    df = pd.read_csv(path)

    # Drop unnamed index column if present
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    before = len(df)
    # Drop rows where 'text' is null or empty
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    after = len(df)

    if before != after:
        print(f"  [INFO] Dropped {before - after:,} rows with missing/empty 'text'.")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main — verification on first 100 rows
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  Step 2 — Text Preprocessing Verification")
    print("=" * 65)

    # Load dataset
    print("\n  Loading dataset...")
    df = load_data()
    print(f"  Total articles loaded : {len(df):,}")
    print(f"  Label distribution    : {df['label'].value_counts().to_dict()}")

    # Work on a small sample
    sample = df.head(100).copy()
    print(f"\n  Applying clean_text() to first {len(sample)} rows...")

    sample["clean_text"] = sample["text"].apply(clean_text)

    print(f"  Done!\n")

    # -----------------------------------------------------------------------
    # Before / After comparison — pick row 0 for the demo
    # -----------------------------------------------------------------------
    idx = 0
    raw  = sample.loc[idx, "text"]
    clean = sample.loc[idx, "clean_text"]
    label = "FAKE" if sample.loc[idx, "label"] == 1 else "REAL"

    print("-" * 65)
    print(f"  ARTICLE #{idx}  |  Label: {label}")
    print("-" * 65)

    # Show first 400 chars of raw text
    print("\n  BEFORE (raw) :")
    print(f"  {raw[:400].strip()!r}")

    # Show first 400 chars of cleaned text
    print("\n  AFTER (cleaned + stemmed) :")
    print(f"  {clean[:400].strip()!r}")

    print("\n" + "-" * 65)
    print(f"  Raw word count     : {len(raw.split()):>6,}")
    print(f"  Cleaned word count : {len(clean.split()):>6,}")
    reduction = (1 - len(clean.split()) / max(len(raw.split()), 1)) * 100
    print(f"  Vocabulary reduced : {reduction:.1f}%")
    print("-" * 65)
    print("\n  Preprocessing pipeline verified successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
