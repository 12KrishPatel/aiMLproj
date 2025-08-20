import streamlit as st
import joblib
import pandas as pd

@st.cache_resource

# Get the model and vectorizer
def load_resources():
    try:
        vec = joblib.load('tfidf_vec.pkl')
        model = joblib.load('spam_detector.pkl')
        return vec, model
    except Exception as e:
        st.error("Error loading resources: {e}")
        st.stop()
tfidf_vec, model_text = load_resources()

st.title("⚠️Spam Detector⚠️")
st.write("⬇Enter the message of your email below to check if its spam or not!⬇")

# Text input area
user_inp = st.text_area("Enter the contents of your email here: ", height=200, placeholder="Type or paste an email here...")

if st.button("Predict"):
    if user_inp:
        # Prep input
        text_series = pd.Series([user_inp])
        text_vec = tfidf_vec.transform(text_series)

        # Predict
        prediction = model_text.predict(text_vec)[0]
        prob = model_text.predict_proba(text_vec)[0]
        spam_confidence = prob[1]*100

        st.subheader("Prediction: ")
        if prediction == 1:
            st.error(f"🛑SPAM🛑 Confidence: {spam_confidence:.2f}%")
        else:
            st.success(f"✅NOT SPAM✅ Confidence: {(100-spam_confidence):.2f}%")
    else:
        st.warning("Please enter some text")

st.markdown("---")
