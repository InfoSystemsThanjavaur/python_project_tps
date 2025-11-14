import streamlit as st
import joblib
import re
import string

# --------------------------
# ✅ 1. Set Streamlit page config (MUST BE FIRST)
# --------------------------
st.set_page_config(page_title="Amazon Sentiment Analyzer", layout="centered")

# --------------------------
# 2. Load models and vectorizer
# --------------------------
@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    nb = joblib.load("models/naive_bayes_model.joblib")
    svm = joblib.load("models/svm_model.joblib")
    return tfidf, nb, svm

tfidf, nb, svm = load_models()

# --------------------------
# 3. Text cleaning function
# --------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------
# 4. Streamlit UI
# --------------------------
st.title("🧠 Sentiment Analysis on Amazon Reviews")
st.markdown("Compare **Naive Bayes** and **SVM** predictions for any review.")

review = st.text_area("Enter a product review below:", height=150)

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter some review text.")
    else:
        # Clean and transform text
        cleaned = clean_text(review)
        X = tfidf.transform([cleaned])

        # Predict using both models
        nb_pred = nb.predict(X)[0]
        svm_pred = svm.predict(X)[0]

        # Convert to readable labels
        nb_label = "Positive 😀" if nb_pred == 1 else "Negative 😠"
        svm_label = "Positive 😀" if svm_pred == 1 else "Negative 😠"

        st.subheader("🧹 Cleaned Text:")
        st.write(cleaned)

        st.subheader("🔮 Predictions:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Naive Bayes", value=nb_label)
        with col2:
            st.metric(label="SVM", value=svm_label)
