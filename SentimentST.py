# SentimentAnalysisApp.py
import os
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from wordcloud import WordCloud

# ---------- NLTK setup ----------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------- CONFIG: set correct paths ----------
MODEL_PATH = r"C:\AnalysisNLPSentiment\env\Scripts\sentiment_pipeline_75plus_nb_balanced.joblib"
PROCESSED_CSV = r"C:\AnalysisNLPSentiment\env\Scripts\SentimentDataset.csv"

# ---------- CUSTOM TEXT PREPROCESSOR ----------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X, y=None):
        return [self._clean_text(text) for text in X]

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s!?]", "", text)  # keep ! and ?
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]
        return " ".join(tokens)

# ---------- HELPERS ----------
@st.cache_data
def safe_load_model(path):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        mdl = joblib.load(path)
        return mdl, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

@st.cache_data
def safe_load_csv(path):
    if not os.path.exists(path):
        return None, f"CSV file not found at: {path}"
    try:
        df = pd.read_csv(path)
        return df, None
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

# Optional: VADER auto-label
def try_auto_label_with_vader(df, text_col='review_text'):
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except Exception:
        return df, "NLTK or VADER not installed."

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()

    def label_text(t):
        if pd.isna(t) or str(t).strip() == "":
            return "neutral"
        s = sia.polarity_scores(str(t))["compound"]
        if s >= 0.05:
            return "positive"
        elif s <= -0.05:
            return "negative"
        else:
            return "neutral"

    df['sentiment'] = df[text_col].fillna("").astype(str).apply(label_text)
    return df, None

# ---------- LOAD model + data ----------
model, model_err = safe_load_model(MODEL_PATH)
df, csv_err = safe_load_csv(PROCESSED_CSV)

st.title("🤖 ChatGPT Reviews — Sentiment Analysis Dashboard")

# show load errors up front
if model_err:
    st.error(model_err)
if csv_err:
    st.error(csv_err)
if df is None:
    st.stop()

# Ensure 'review_text' exists
if 'review_text' not in df.columns:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ('review', 'text', 'comment'))]
    if candidates:
        df['review_text'] = df[candidates[0]].astype(str)
    elif df.shape[1] == 1:
        df['review_text'] = df.iloc[:, 0].astype(str)
    else:
        st.error("No 'review_text' column found and no obvious fallback column.")
        st.stop()

df['review_text'] = df['review_text'].astype(str)

# Auto-label sentiment if missing
if 'sentiment' not in df.columns:
    df, msg = try_auto_label_with_vader(df, text_col='review_text')
    if msg:
        st.warning("Sentiment column missing and VADER labeling failed: " + msg)
        st.info("Some visualizations may not work.")
    else:
        st.info("Auto-labeled 'sentiment' using VADER.")

# Fallback if sentiment still missing
if 'sentiment' not in df.columns:
    df['sentiment'] = "neutral"

# Drop empty reviews
df = df[df['review_text'].str.strip().astype(bool)].reset_index(drop=True)
if df.empty:
    st.warning("No non-empty reviews found.")
    st.stop()

# ----------------- SENTIMENT DISTRIBUTION -----------------
st.subheader("Overall Sentiment Distribution")
if 'sentiment' in df.columns:
    sentiment_dist = df['sentiment'].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(sentiment_dist)

# ----------------- RATING DISTRIBUTION -----------------
if 'rating' in df.columns:
    st.subheader("Rating Distribution (1 to 5 Stars)")
    try:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        st.bar_chart(df['rating'].value_counts().sort_index())
    except Exception:
        st.write("Rating column cannot be converted to numeric.")

# ----------------- HELPFUL VOTES FILTER -----------------
if 'helpful_votes' in df.columns:
    st.subheader("Helpful Reviews Filter")
    try:
        df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0).astype(int)
        threshold = st.slider("Minimum helpful votes:", 0, int(df['helpful_votes'].max()), 5)
        st.write("Number of reviews above threshold:", int((df['helpful_votes'] > threshold).sum()))
    except Exception:
        st.write("helpful_votes column cannot be parsed.")

# ----------------- WORD CLOUDS -----------------
st.subheader("Word Clouds (Positive vs Negative)")
col1, col2 = st.columns(2)
for col, sentiment in zip([col1, col2], ["positive", "negative"]):
    texts = df.loc[df['sentiment'] == sentiment, 'review_text'].astype(str).tolist()
    text = " ".join(texts)
    if text.strip():
        wc = WordCloud(width=500, height=300, background_color="white").generate(text)
        col.image(wc.to_array(), use_container_width=True, caption=f"{sentiment.capitalize()} Reviews")
    else:
        col.write(f"No {sentiment} reviews available.")

# ----------------- SENTIMENT OVER TIME -----------------
if 'date' in df.columns:
    st.subheader("Sentiment Trend Over Time (Monthly)")
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df_time = df.dropna(subset=['date_parsed']).copy()
    if not df_time.empty:
        df_time['month'] = df_time['date_parsed'].dt.to_period('M')
        counts = df_time.groupby(['month', 'sentiment']).size().reset_index(name='count')
        pivot = counts.pivot(index='month', columns='sentiment', values='count').fillna(0).sort_index()
        monthly = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        if hasattr(monthly.index, "to_timestamp"):
            monthly.index = monthly.index.to_timestamp()
        else:
            monthly.index = pd.to_datetime(monthly.index.astype(str), errors='coerce')
        if not monthly.empty:
            st.line_chart(monthly)

# ----------------- VERIFIED PURCHASE COMPARISON -----------------
if 'verified_purchase' in df.columns:
    st.subheader("Verified vs Non-Verified Sentiment")
    try:
        table = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index').fillna(0)
        st.dataframe(table)
    except Exception as e:
        st.write("Could not compute verified vs sentiment: ", e)

# ----------------- PLATFORM SENTIMENT -----------------
if 'platform' in df.columns:
    st.subheader("Sentiment by Platform (Web vs Mobile)")
    try:
        table = pd.crosstab(df['platform'], df['sentiment'], normalize='index').fillna(0)
        st.dataframe(table)
    except Exception as e:
        st.write("Could not compute platform vs sentiment: ", e)

# ----------------- LIVE PREDICTION (Plug-and-Play) -----------------
st.subheader("Predict Sentiment for New Review")
new_text = st.text_area("Type a review:")

def predict_sentiment(text):
    if model is None:
        return None, "Model not loaded."
    if not text.strip():
        return None, "Please enter text before predicting."
    try:
        # Automatically preprocess inside model if pipeline includes TextPreprocessor
        pred = model.predict([text])[0]
        return pred, None
    except Exception as e:
        return None, f"Prediction failed: {e}"

if st.button("Predict"):
    pred, error = predict_sentiment(new_text)
    if error:
        st.warning(error)
    else:
        st.success(f"Predicted sentiment: **{pred}**")

st.markdown("---")
st.caption("📦 Model & processed dataset loaded from provided local files.")
