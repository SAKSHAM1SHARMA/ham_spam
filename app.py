import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load vectorizer and model
vectorizer = joblib.load('count_vectorizer.pkl')
model = joblib.load('spam_classifier_model.pkl')

# Streamlit UI
st.set_page_config(page_title="Spam Detection", layout="centered")

st.title("📩 Spam Message Classifier")
st.markdown("Enter a message below to classify it as **Spam** or **Ham**:")

text_input = st.text_area("Enter Message:", height=200)

if st.button("Classify", type="primary"):
    if text_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        text_vectorized = vectorizer.transform([text_input])
        prediction = model.predict(text_vectorized)[0]
        if prediction == 1:
            st.error("🚨 This message is classified as **Spam**.")
        else:
            st.success("✅ This message is classified as **Ham**.")

