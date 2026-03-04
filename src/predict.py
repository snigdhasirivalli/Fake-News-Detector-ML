"""
Quick smoke-test for the trained Fake News Detection model.
Run:  python src/predict.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import joblib
from src.preprocess import clean_text

MODEL_PATH      = pathlib.Path("models/fake_news_model.pkl")
VECTORIZER_PATH = pathlib.Path("models/tfidf_vectorizer.pkl")

# Load saved artifacts
model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- Test articles ---
test_articles = [
    {
        "label": "REAL",
        "text": (
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
            "citing continued progress toward its 2% inflation target. Fed Chair Jerome Powell "
            "said the central bank would remain data-dependent in future decisions, emphasizing "
            "that officials are carefully monitoring both inflation and labor market conditions."
        ),
    },
    {
        "label": "FAKE",
        "text": (
            "BREAKING: Scientists confirm that drinking bleach cures all known diseases! "
            "The government has been hiding this miracle cure for decades to protect Big Pharma profits. "
            "Share this before they DELETE it! Obama and Hillary are behind the cover-up, "
            "insiders reveal shocking truth about deep state conspiracy against the American people!!"
        ),
    },
    {
        "label": "REAL",
        "text": (
            "NASA's Perseverance rover has collected its 20th rock sample on Mars, "
            "advancing scientists' understanding of the planet's geological history. "
            "The samples will be returned to Earth as part of the Mars Sample Return mission, "
            "a joint effort between NASA and the European Space Agency expected to launch in 2028."
        ),
    },
    {
        "label": "FAKE",
        "text": (
            "EXCLUSIVE: Microchips found in COVID vaccines activate 5G towers to control human thoughts! "
            "A whistleblower doctor has leaked documents proving the vaccines contain nano-robots "
            "programmed by globalists. The mainstream media is SILENCING this story. "
            "WAKE UP SHEEPLE and share before the New World Order deletes this!!!"
        ),
    },
]

print("=" * 65)
print("  Model Smoke Test — Live Predictions")
print("=" * 65)

correct = 0
for i, article in enumerate(test_articles):
    cleaned   = clean_text(article["text"])
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    pred_label = "FAKE" if prediction == 1 else "REAL"
    actual     = article["label"]
    match      = pred_label == actual

    if match:
        correct += 1

    status = "[PASS]" if match else "[FAIL]"
    icon   = "+"      if match else "X"

    print(f"\n  [{i+1}] {icon} {status}")
    print(f"      Actual    : {actual}")
    print(f"      Predicted : {pred_label}")
    print(f"      Text      : {article['text'][:80].strip()}...")

print("\n" + "-" * 65)
print(f"  Result  : {correct}/{len(test_articles)} correct  ({correct/len(test_articles)*100:.0f}%)")
print(f"  Status  : {'ALL PASSED - Model is working!' if correct == len(test_articles) else 'Some predictions were wrong.'}")
print("=" * 65)
