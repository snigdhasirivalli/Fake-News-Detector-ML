"""
src/train_model.py
------------------
Step 3 — TF-IDF Vectorization + Model Training Pipeline

Pipeline:
  1. Load & clean the full WELFake dataset (via src.preprocess)
  2. TF-IDF vectorization  (max_features=10000, ngram_range=(1,2))
  3. 80/20 train-test split  (stratified)
  4. LogisticRegression  — better calibration & real-world generalization
     than PassiveAggressiveClassifier on out-of-distribution text
  5. Evaluate  →  Accuracy + full Classification Report
  6. Save model  →  models/fake_news_model.pkl
     Save vectorizer  →  models/tfidf_vectorizer.pkl

Usage:
    python src/train_model.py
"""

import time
import pathlib
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
PROJECT_ROOT  = pathlib.Path(__file__).parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
MODEL_PATH    = MODELS_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH  = MODELS_DIR / "metrics.txt"

MODELS_DIR.mkdir(exist_ok=True)

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

    _section("Step 3 — TF-IDF Vectorization & Model Training")

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    _step("Loading dataset...")
    df = load_data()
    print(f"     Articles loaded  : {len(df):,}")
    print(f"     Fake (1)         : {(df['label']==1).sum():,}")
    print(f"     Real (0)         : {(df['label']==0).sum():,}")

    _step("Cleaning text (this may take 2-4 minutes for 71k rows)...")
    t0 = time.perf_counter()
    df["clean_content"] = df["content"].apply(clean_text)
    elapsed = time.perf_counter() - t0
    print(f"     Cleaning done in {elapsed:.1f}s")

    X = df["clean_content"]
    y = df["label"]

    # ------------------------------------------------------------------
    # 2. Train / Test Split  (80 / 20, stratified to keep label balance)
    # ------------------------------------------------------------------
    _step("Splitting data  →  80% train / 20% test  (stratified)")
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
    # ------------------------------------------------------------------
    _step("Fitting TF-IDF vectorizer  (max_features=10000, ngram_range=(1,2))...")
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=10000,       # increased from 5k for richer vocabulary
        ngram_range=(1, 2),
        sublinear_tf=True,        # apply log normalization to TF
        min_df=2,                 # ignore terms that appear in < 2 docs
        strip_accents="unicode",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    elapsed = time.perf_counter() - t0
    print(f"     Vectorizer fitted in {elapsed:.2f}s")
    print(f"     Vocabulary size  : {len(vectorizer.vocabulary_):,} features")
    print(f"     Train matrix     : {X_train_tfidf.shape}")
    print(f"     Test  matrix     : {X_test_tfidf.shape}")

    # ------------------------------------------------------------------
    # 4. Train — LogisticRegression
    #    Chosen over PassiveAggressiveClassifier because it:
    #    - Generalizes better to real-world, out-of-distribution news
    #    - Provides well-calibrated probabilities
    #    - Is less sensitive to the specific writing style in training data
    # ------------------------------------------------------------------
    _step("Training LogisticRegression (this may take ~1 minute)...")
    t0 = time.perf_counter()
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=5.0,                    # regularization (higher = less regularized)
        solver="lbfgs",
        n_jobs=-1,                # use all CPU cores
    )
    model.fit(X_train_tfidf, y_train)
    elapsed = time.perf_counter() - t0
    print(f"     Model trained in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate
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

    print(f"\n  {'─'*63}")
    print(f"  {'EVALUATION RESULTS':^63}")
    print(f"  {'─'*63}")
    print(f"\n  Accuracy Score : {accuracy * 100:.2f}%\n")
    print("  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':12}  Pred Real  Pred Fake")
    print(f"    Actual Real   {cm[0][0]:>7,}    {cm[0][1]:>7,}")
    print(f"    Actual Fake   {cm[1][0]:>7,}    {cm[1][1]:>7,}")
    print(f"  {'─'*63}")

    # ------------------------------------------------------------------
    # 6. Save model artifacts
    # ------------------------------------------------------------------
    _step("Saving model artifacts...")

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Also write a plain-text metrics summary
    metrics_text = (
        f"Fake News Detector — Model Metrics\n"
        f"{'='*40}\n"
        f"Algorithm   : LogisticRegression\n"
        f"Vectorizer  : TF-IDF (max_features=10000, ngram=(1,2))\n"
        f"Train rows  : {len(X_train):,}\n"
        f"Test  rows  : {len(X_test):,}\n"
        f"Accuracy    : {accuracy * 100:.2f}%\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    METRICS_PATH.write_text(metrics_text, encoding="utf-8")

    print(f"\n  {'─'*63}")
    print(f"  Artifacts saved successfully:")
    print(f"    Model      →  {MODEL_PATH}")
    print(f"    Vectorizer →  {VECTORIZER_PATH}")
    print(f"    Metrics    →  {METRICS_PATH}")
    print(f"  {'─'*63}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Total pipeline completed in {total_elapsed:.1f}s")
    _section("Training Complete!")


if __name__ == "__main__":
    main()
