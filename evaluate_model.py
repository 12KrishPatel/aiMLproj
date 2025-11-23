"""
Model Evaluation Script for Spam Detector
Quantifies model performance metrics without modifying existing code
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SPAM DETECTOR MODEL EVALUATION")
print("=" * 70)

# Load the trained model and vectorizer
print("\n📦 Loading saved model and vectorizer...")
model = joblib.load('spam_detector.pkl')
vectorizer = joblib.load('tfidf_vec.pkl')

# Load datasets (same as training)
print("📊 Loading datasets...")
df1 = pd.read_csv('messages.csv')
df2 = pd.read_csv('phishing_email.csv')

# Preprocess df1
df1.columns = df1.columns.str.strip()
df1['text_combined'] = df1['subject'].fillna('') + ' ' + df1['message'].fillna('')
df1 = df1[['text_combined', 'label']]

# Preprocess df2
df2.columns = df2.columns.str.strip()
df2 = df2[['text_combined', 'label']]

# Balance the dataset
min_samples = min(len(df1), len(df2))
df1_balanced = df1.sample(n=min_samples, random_state=42)
df2_balanced = df2.sample(n=min_samples, random_state=42)
df = pd.concat([df1_balanced, df2_balanced], ignore_index=True)

# Prepare features and labels
X = df['text_combined']
y = df['label']

# Create train/test split (same random_state as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Define handcrafted feature functions (same as training)
def extract_features(text):
    urgent_words = ['urgent', 'immediately', 'now', 'action', 'disabled', 'limited',
                   'suspended', 'expire', 'warning', 'important']
    threat_words = ['permanently', 'terminated', 'disabled', 'sorry', 'inform',
                   'no longer have access']
    link_words = ['click', 'link', 'verify', 'confirm', 'login', 'secure', 'account details']

    text_lower = text.lower() if isinstance(text, str) else ''

    return [
        len(text_lower),
        sum(1 for word in urgent_words if word in text_lower),
        sum(1 for word in threat_words if word in text_lower),
        sum(1 for word in link_words if word in text_lower)
    ]

# Transform test data with TF-IDF
print("\n🔧 Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)

# Extract handcrafted features
X_test_handcrafted = np.array([extract_features(text) for text in X_test])

# Combine features
X_test_combined = hstack([X_test_tfidf, X_test_handcrafted])

print(f"   Feature dimensions: {X_test_combined.shape}")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred = model.predict(X_test_combined)
y_pred_proba = model.predict_proba(X_test_combined)[:, 1]

# Calculate metrics
print("\n" + "=" * 70)
print("📊 MODEL PERFORMANCE METRICS")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"📈 Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"⚖️  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"📉 ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# Confusion Matrix
print("\n" + "=" * 70)
print("📋 CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test, y_pred)
print(f"\n                Predicted")
print(f"              Not Spam  |  Spam")
print(f"           ----------------------")
print(f"Not Spam  |   {cm[0][0]:6d}   |  {cm[0][1]:6d}")
print(f"Spam      |   {cm[1][0]:6d}   |  {cm[1][1]:6d}")

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")
print(f"Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")

# Detailed Classification Report
print("\n" + "=" * 70)
print("📄 DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'], digits=4))

# Model Information
print("=" * 70)
print("🤖 MODEL INFORMATION")
print("=" * 70)
print(f"\nModel Type: Random Forest Classifier")
print(f"Number of Trees: {model.n_estimators}")
print(f"Total Features: {X_test_combined.shape[1]}")
print(f"  - TF-IDF Features: {X_test_tfidf.shape[1]}")
print(f"  - Handcrafted Features: {X_test_handcrafted.shape[1]}")
print(f"Class Balance: {sum(y_test == 0)} Not Spam, {sum(y_test == 1)} Spam")

# Summary for Resume
print("\n" + "=" * 70)
print("📝 RESUME-READY SUMMARY")
print("=" * 70)
print(f"""
Developed a spam/phishing email detection system using Random Forest:
• Achieved {accuracy*100:.2f}% accuracy on {len(X_test):,} test samples
• Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1-Score: {f1*100:.2f}%
• ROC-AUC Score: {roc_auc*100:.2f}%
• Engineered {X_test_combined.shape[1]:,} features (TF-IDF + domain-specific features)
• Trained on balanced dataset with {len(X_train):,} samples
• Deployed via interactive Streamlit web application
""")

print("=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)
