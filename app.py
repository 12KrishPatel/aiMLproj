import streamlit as st
import joblib
import pandas as pd
import scipy.sparse

# Define keyword lists (same as in training)
urgent_words = ['urgent', 'immediately', 'now', 'action', 'disabled', 'limited', 'suspended', 'expire', 'warning', 'important']
threat_words = ['permanently', 'terminated', 'disabled', 'sorry', 'inform', 'no longer have access']
link_words = ['click', 'link', 'verify', 'confirm', 'login', 'secure', 'account details']

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

def extract_features(text):
    """Extract handcrafted features from text (same as training)"""
    length = len(text)
    urgent_count = sum(1 for word in urgent_words if word in text.lower())
    threat_count = sum(1 for word in threat_words if word in text.lower())
    link_count = sum(1 for word in link_words if word in text.lower())
    return [length, urgent_count, threat_count, link_count]

st.title("⚠️Spam Detector⚠️")
st.write("⬇️Enter the message of your email below to check if its spam.⬇️")

# Text input area
user_inp = st.text_area("Enter the contents of your email here: ", height=200, placeholder="Type or paste an email here...")

if st.button("Predict"):
    if user_inp:
        # Prep input - TF-IDF features
        text_series = pd.Series([user_inp])
        text_vec = tfidf_vec.transform(text_series)

        # Extract handcrafted features
        extra_features = extract_features(user_inp)
        extra_features_array = pd.DataFrame([extra_features], columns=['length', 'urgent_count', 'threat_count', 'link_count'])

        # Combine TF-IDF features with handcrafted features (same as training)
        combined_features = scipy.sparse.hstack((text_vec, extra_features_array.values))

        # Predict
        prediction = model_text.predict(combined_features)[0]
        prob = model_text.predict_proba(combined_features)[0]
        spam_confidence = prob[1]*100

        st.subheader("Prediction: ")
        if prediction == 1:
            st.error(f"🛑SPAM🛑 Confidence: {spam_confidence:.2f}%")
        else:
            st.success(f"✅NOT SPAM✅ Confidence: {(100-spam_confidence):.2f}%")
    else:
        st.warning("Please enter some text")

st.markdown("---")
