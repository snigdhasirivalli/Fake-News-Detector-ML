"""
src/train_model.py
------------------
Step 3 -- TF-IDF Vectorization + Model Training Pipeline

Pipeline:
  1. Load & clean the full WELFake dataset (via src.preprocess)
  2. Strip dataset-specific watermarks from the RAW TEXT before cleaning
  3. TF-IDF vectorization  (max_features=10000, ngram_range=(1,2))
  4. 80/20 train-test split  (stratified)
  5. LogisticRegression with CalibratedClassifierCV for well-calibrated probabilities
  6. Evaluate  ->  Accuracy + full Classification Report
  7. Save model  ->  models/fake_news_model.pkl
     Save vectorizer  ->  models/tfidf_vectorizer.pkl

Usage:
    python src/train_model.py
"""

import re
import time
import pathlib
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Import our own preprocessing helpers
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.preprocess import clean_text, load_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = pathlib.Path(__file__).parent.parent
MODELS_DIR      = PROJECT_ROOT / "models"
MODEL_PATH      = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH    = MODELS_DIR / "metrics.txt"

MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset-specific watermarks to strip from RAW text before any ML touches it.
# These terms appear almost exclusively in one class in the WELFake dataset
# and cause the model to learn the *source* of an article rather than its
# writing style / content quality.
# ---------------------------------------------------------------------------
_WATERMARK_PATTERN = re.compile(
    r"\b("
    r"reuters?|breitbart|getty\s*images?|associated\s*press|"
    r"featured?\s*image|read\s*more|follow\s+us\s+on|"
    r"follow\s+on\s+twitter|via\s+breitbart|gettyi|"
    r"photo\s*credit|image\s*credit|watch\s*video|"
    r"click\s*here\s*to\s*read"
    r")\b",
    re.IGNORECASE,
)

def strip_watermarks(text: str) -> str:
    """Remove dataset-specific publication watermarks from raw text."""
    if not isinstance(text, str):
        return ""
    return _WATERMARK_PATTERN.sub(" ", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def _step(msg: str):
    print(f"\n  >> {msg}")


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------
def main():
    total_start = time.perf_counter()

    _section("Step 3 -- TF-IDF Vectorization & Model Training")

    # ------------------------------------------------------------------
    # 1. Load & clean data  (with parquet cache for fast reruns)
    # ------------------------------------------------------------------
    CACHE_PATH = PROJECT_ROOT / "data" / "cleaned_cache.parquet"

    if CACHE_PATH.exists():
        _step("Loading from cache (skipping cleaning step)...")
        cache_df = __import__('pandas').read_parquet(CACHE_PATH)
        X = cache_df["clean_content"]
        y = cache_df["label"]
        print(f"     Loaded {len(X):,} cached rows from {CACHE_PATH.name}")
    else:
        _step("Loading dataset...")
        df = load_data()
        print(f"     Articles loaded  : {len(df):,}")
        print(f"     Fake (1)         : {(df['label']==1).sum():,}")
        print(f"     Real (0)         : {(df['label']==0).sum():,}")

        _step("Stripping publication watermarks from raw text...")
        df["content"] = df["content"].apply(strip_watermarks)

        _step("Cleaning text (this may take 10+ minutes for 71k rows)...")
        t0 = time.perf_counter()
        df["clean_content"] = df["content"].apply(clean_text)
        elapsed = time.perf_counter() - t0
        print(f"     Cleaning done in {elapsed:.1f}s")

        # Save cache so next run is instant
        _step("Saving cleaned data cache...")
        df[["clean_content", "label"]].to_parquet(CACHE_PATH, index=False)
        print(f"     Cache saved to {CACHE_PATH.name}")

        X = df["clean_content"]
        y = df["label"]

    # ------------------------------------------------------------------
    # 2. Train / Test Split  (80 / 20, stratified to keep label balance)
    # ------------------------------------------------------------------
    _step("Splitting data  ->  80% train / 20% test  (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"     Train samples : {len(X_train):,}")
    print(f"     Test  samples : {len(X_test):,}")

    # ------------------------------------------------------------------
    # 3. TF-IDF Vectorization
    #    - NO custom stop_words here -- watermarks already stripped above
    #    - char_wb analyzer adds character n-grams to capture style signals
    # ------------------------------------------------------------------
    _step("Fitting TF-IDF vectorizer  (max_features=10000, ngram_range=(1,2))...")
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,        # log normalization keeps high-freq words from dominating
        min_df=3,                 # ignore very rare terms (likely noise)
        max_df=0.95,              # ignore terms in >95% of docs (near-universal, uninformative)
        strip_accents="unicode",
        token_pattern=r"\b[a-z][a-z]+\b",  # words of 2+ chars only
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    elapsed = time.perf_counter() - t0
    print(f"     Vectorizer fitted in {elapsed:.2f}s")
    print(f"     Vocabulary size  : {len(vectorizer.vocabulary_):,} features")
    print(f"     Train matrix     : {X_train_tfidf.shape}")
    print(f"     Test  matrix     : {X_test_tfidf.shape}")

    # ------------------------------------------------------------------
    # 4. Train -- LogisticRegression
    #    - fit_intercept=True  (RESTORED -- allows the model to learn a
    #      proper baseline instead of defaulting to the majority class)
    #    - C=1.0  (standard regularization -- prevents overfitting to
    #      dataset-specific vocabulary)
    #    - class_weight='balanced'  (handles any label imbalance)
    # ------------------------------------------------------------------
    _step("Training LogisticRegression (this may take ~1 minute)...")
    t0 = time.perf_counter()
    base_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,      # RESTORED: lets model learn proper baseline
        solver="lbfgs",
        n_jobs=-1,
    )
    base_model.fit(X_train_tfidf, y_train)
    elapsed = time.perf_counter() - t0
    print(f"     Base model trained in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 5. Calibrate probabilities using isotonic regression
    #    This ensures predict_proba() gives reliable confidence scores
    #    rather than overconfident outputs from the raw logistic model.
    # ------------------------------------------------------------------
    _step("Calibrating probability outputs (isotonic regression, cv=3)...")
    t0 = time.perf_counter()
    model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    model.fit(X_train_tfidf, y_train)
    elapsed = time.perf_counter() - t0
    print(f"     Calibration done in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    _step("Evaluating on test set...")
    y_pred    = model.predict(X_test_tfidf)
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=["Real (0)", "Fake (1)"],
        digits=4,
    )
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  {'-'*63}")
    print(f"  {'EVALUATION RESULTS':^63}")
    print(f"  {'-'*63}")
    print(f"\n  Accuracy Score : {accuracy * 100:.2f}%\n")
    print("  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':12}  Pred Real  Pred Fake")
    print(f"    Actual Real   {cm[0][0]:>7,}    {cm[0][1]:>7,}")
    print(f"    Actual Fake   {cm[1][0]:>7,}    {cm[1][1]:>7,}")
    print(f"  {'-'*63}")

    # ------------------------------------------------------------------
    # 7. Spot-check with known real news headlines to verify bias is gone
    # ------------------------------------------------------------------
    _step("Spot-checking with known real news text...")
    test_cases = [
        ("REAL",
         "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
         "citing continued progress toward its 2% inflation target. Fed Chair Jerome Powell "
         "said the committee remains committed to returning inflation to the 2% goal."),
        ("REAL",
         "NASA successfully launched the Artemis mission to the Moon on Saturday. "
         "The rocket carrying the Orion spacecraft lifted off from Kennedy Space Center "
         "in Florida at 1:47 AM EST, marking a major milestone for human space exploration."),
        ("FAKE",
         "SHOCKING: Government secretly putting chemicals in water supply to control minds! "
         "Scientists CONFIRM what they don't want you to know. Share this before it gets deleted! "
         "Deep state exposed - wake up America! Click here to see the proof they're hiding from you."),
    ]
    print()
    for expected, text in test_cases:
        cleaned = clean_text(strip_watermarks(text))
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        pred = "FAKE" if proba[1] >= 0.5 else "REAL"
        conf = max(proba) * 100
        status = "[OK]" if pred == expected else "[WRONG] WRONG"
        print(f"     [{status}] Expected={expected}, Got={pred} ({conf:.1f}% conf)")
        print(f"            Text: {text[:80]}...")

    # ------------------------------------------------------------------
    # 8. Save model artifacts
    # ------------------------------------------------------------------
    _step("Saving model artifacts...")

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Also write a plain-text metrics summary
    metrics_text = (
        f"Fake News Detector -- Model Metrics\n"
        f"{'='*40}\n"
        f"Algorithm   : LogisticRegression + CalibratedClassifierCV (isotonic)\n"
        f"Vectorizer  : TF-IDF (max_features=10000, ngram=(1,2))\n"
        f"Watermarks  : Stripped from raw text before cleaning\n"
        f"Train rows  : {len(X_train):,}\n"
        f"Test  rows  : {len(X_test):,}\n"
        f"Accuracy    : {accuracy * 100:.2f}%\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    print(f"\n  {'-'*63}")
    print(f"  Artifacts saved successfully:")
    print(f"    Model      ->  {MODEL_PATH}")
    print(f"    Vectorizer ->  {VECTORIZER_PATH}")
    print(f"    Metrics    ->  {METRICS_PATH}")
    print(f"  {'-'*63}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Total pipeline completed in {total_elapsed:.1f}s")
    _section("Training Complete!")


if __name__ == "__main__":
    main()
