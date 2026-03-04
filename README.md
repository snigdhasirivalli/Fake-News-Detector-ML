# 📰 Fake News Detector — ML Project

A machine-learning pipeline that classifies news articles as **Real** or **Fake**
using the [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
(72,134 articles, balanced 50/50).

---

## 🗂 Project Structure

```
fake_news_detection/
│
├── data/               ← Place WELFake_Dataset.csv here (not tracked by git)
├── models/             ← Saved trained models (.pkl / .joblib)
├── notebooks/          ← Exploration & training scripts
│   └── 01_load_dataset.py
│
├── verify_setup.py     ← Confirm all dependencies are installed
├── requirements.txt    ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Fake-News-Detector-ML.git
cd Fake-News-Detector-ML
```

### 2. Create & activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify the environment
```bash
python verify_setup.py
```

### 5. Download the dataset
1. Go to → [WELFake Dataset on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
2. Download `WELFake_Dataset.csv`
3. Place it inside the `data/` folder

### 6. Preview the dataset
```bash
python notebooks/01_load_dataset.py
```

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `scikit-learn` | ML models, TF-IDF, evaluation |
| `nltk` | Text preprocessing (stopwords, stemming) |
| `streamlit` | Interactive web demo |
| `joblib` | Model serialization |

---

## 🗺 Roadmap

- [x] Step 1 — Environment & dataset setup
- [ ] Step 2 — Data cleaning & EDA
- [ ] Step 3 — Feature engineering (TF-IDF)
- [ ] Step 4 — Model training & evaluation
- [ ] Step 5 — Streamlit app deployment
