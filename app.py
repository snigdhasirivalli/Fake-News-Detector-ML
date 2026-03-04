"""
app.py
------
Streamlit web application for the AI Fake News Detector.
Run with:  streamlit run app.py
"""

import re
import time
import pathlib

import joblib
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Text cleaning  (self-contained — no external import needed)
# ─────────────────────────────────────────────────────────────────────────────
import nltk

@st.cache_resource(show_spinner=False)
def _load_nltk():
    for path, pkg in [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    return set(stopwords.words("english")), PorterStemmer()

_STOP_WORDS, _STEMMER = _load_nltk()

def clean_text(text: str) -> str:
    """Full NLP cleaning pipeline (lowercase → regex → tokenize → stem)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    tokens = [_STEMMER.stem(t) for t in tokens
              if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load model artifacts  (cached so they load only once)
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = pathlib.Path(__file__).parent / "models"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model      = joblib.load(MODELS_DIR / "fake_news_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Custom CSS  — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root / Background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* ── Text area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 1rem !important;
    transition: border-color 0.25s;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.8) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}
.stTextArea label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.55) !important;
    background: linear-gradient(135deg, #6d28d9, #2563eb) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result boxes ── */
.result-fake {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.45);
    border-left: 4px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    animation: slideUp 0.4s ease;
}
.result-real {
    background: rgba(52,211,153,0.10);
    border: 1px solid rgba(52,211,153,0.45);
    border-left: 4px solid #34d399;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    animation: slideUp 0.4s ease;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.result-fake .result-title  { color: #fca5a5; }
.result-real .result-title  { color: #6ee7b7; }
.result-subtitle {
    font-size: 0.92rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Confidence bar ── */
.conf-label {
    display: flex;
    justify-content: space-between;
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 1.1rem 0 0.3rem;
}
.conf-track {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.conf-fill-fake {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f87171, #ef4444);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.conf-fill-real {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #34d399, #10b981);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── Stats pills ── */
.pill-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: #cbd5e1;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
}

/* ── Slide-up animation ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── How it works section ── */
.how-title {
    color: #94a3b8;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.8rem;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.65rem;
}
.step-num {
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    min-width: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}
.step-text { color: #94a3b8; font-size: 0.88rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛡️ AI Fake News Detector</h1>
    <p>Paste any news article below and let the model analyze it in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Input card ───────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

news_input = st.text_area(
    label="📰 Paste your news article here",
    placeholder=(
        "e.g.  'The Federal Reserve raised interest rates by 25 basis points "
        "on Wednesday, citing continued progress toward its 2% inflation target…'"
    ),
    height=220,
    key="news_input",
)

word_count = len(news_input.split()) if news_input.strip() else 0
char_count = len(news_input)

if news_input.strip():
    st.markdown(
        f'<div class="pill-row">'
        f'<span class="pill">📝 {word_count} words</span>'
        f'<span class="pill">🔤 {char_count} characters</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Check Authenticity", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
if predict_clicked:
    if not news_input.strip():
        st.warning("⚠️  Please paste some news text before clicking the button.")
    elif word_count < 10:
        st.warning("⚠️  Please provide at least 10 words for a reliable prediction.")
    else:
        with st.spinner("Analyzing article…"):
            time.sleep(0.6)   # small delay so the spinner is visible

            cleaned    = clean_text(news_input)
            vectorized = vectorizer.transform([cleaned])
            
            # Use predict_proba for LogisticRegression to get actual probability
            probabilities = model.predict_proba(vectorized)[0]
            
            # Class 0 = Real, Class 1 = Fake
            prob_fake = probabilities[1]
            prediction = 1 if prob_fake >= 0.5 else 0
            
            # Confidence is the probability of the predicted class
            confidence_raw = prob_fake if prediction == 1 else (1.0 - prob_fake)
            confidence = round(confidence_raw * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-fake">
                <p class="result-title">🚨 This news appears to be FAKE</p>
                <p class="result-subtitle">
                    The model detected patterns commonly associated with misinformation,
                    sensationalism, or fabricated content.
                </p>
                <div class="conf-label">
                    <span>Confidence</span><span>{confidence}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill-fake" style="width:{confidence}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <p class="result-title">✅ This news appears to be REAL</p>
                <p class="result-subtitle">
                    The model found patterns consistent with credible, factual reporting.
                    Always verify with a trusted source.
                </p>
                <div class="conf-label">
                    <span>Confidence</span><span>{confidence}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill-real" style="width:{confidence}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Divider + How it works ────────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="how-title">How it works</p>', unsafe_allow_html=True)
st.markdown("""
<div class="step-row">
    <div class="step-num">1</div>
    <div class="step-text"><b style="color:#e2e8f0">Text Cleaning</b> — Lowercase, remove special characters, strip stopwords, apply Porter Stemming.</div>
</div>
<div class="step-row">
    <div class="step-num">2</div>
    <div class="step-text"><b style="color:#e2e8f0">TF-IDF Vectorization</b> — Converts the cleaned text into a 5,000-feature numerical matrix using unigrams &amp; bigrams.</div>
</div>
<div class="step-row">
    <div class="step-num">3</div>
    <div class="step-text"><b style="color:#e2e8f0">Classification</b> — A PassiveAggressiveClassifier trained on 57,000+ articles classifies the article as Real or Fake.</div>
</div>
<div class="step-row">
    <div class="step-num">4</div>
    <div class="step-text"><b style="color:#e2e8f0">Confidence Score</b> — The model's decision boundary distance is converted to a 0–100% confidence estimate.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<p style="text-align:center; color:#475569; font-size:0.8rem; margin-top:1rem;">
    Trained on the <b>WELFake Dataset</b> · 71,351 articles · 95.79% accuracy  ·
    Built with Streamlit &amp; scikit-learn
</p>
""", unsafe_allow_html=True)
