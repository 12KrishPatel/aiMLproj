import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Ensure messages DF loads
try: 
    messages_df = pd.read_csv('messages.csv')
    phishing_df = pd.read_csv('phishing_email.csv')
except FileNotFoundError:
    print("Error: messages.csv not found")
    exit()

# Fill w/ empty strings to avoid errors
messages_df['subject'] = messages_df['subject'].fillna('')
messages_df['message'] = messages_df['message'].fillna('')

# Combine messages and subjects into one column
messages_df['text'] = messages_df['subject'] + "" + messages_df['message']

# Renaming second datesets columns
phishing_df.rename(columns={'text_combined': 'text', 'label': 'label'}, inplace=True)

# Combine the dataframes
combined_df = pd.concat([messages_df, phishing_df], ignore_index=True)

# Seperating the datasets
df_spam = combined_df[combined_df['label'] == 1]
df_not_spam = combined_df[combined_df['label'] == 0]
min_size = min(len(df_spam), len(df_not_spam))
max_size = max(len(df_spam), len(df_not_spam))
urgent_words = ['urgent', 'immediately', 'now', 'action', 'disabled', 'limited', 'suspended', 'expire', 'warning', 'important']
threat_words = ['permanently', 'terminated', 'disabled', 'sorry', 'inform', 'no longer have access']
link_words = ['click', 'link', 'verify', 'confirm', 'login', 'secure', 'account details']

if(len(df_spam) < len(df_not_spam)):
    from sklearn.utils import resample
    df_spam_resam = resample(df_spam, replace=True, n_samples = max_size, random_state=42)
    balanced_df = pd.concat([df_not_spam, df_spam_resam])
else:
    from sklearn.utils import resample
    df_not_spam_resam = resample(df_not_spam, replace=True, n_samples=min_size, random_state=42)
    balanced_df = pd.concat([df_spam, df_not_spam_resam])

balanced_df['length'] = balanced_df['text'].apply(len)

balanced_df['urgent_count'] = balanced_df['text'].apply(lambda x: sum(1 for word in x.lower()))
balanced_df['threat_count'] = balanced_df['text'].apply(lambda x: sum(1 for word in x.lower()))
balanced_df['link_count'] = balanced_df['text'].apply(lambda x: sum(1 for word in x.lower()))

print("First 5 rows of 'text_content' and 'label':")
print(balanced_df[['text', 'label']].head())

print("\nDistribution of 'label' in new dataset:")
print(balanced_df['label'].value_counts())
print(balanced_df['label'].value_counts(normalize=True))

# Seperate X (features) and Y (target)
xtxt = balanced_df['text']
xtra_features = balanced_df[['length', 'urgent_count', 'threat_count', 'link_count']]
ytxt = balanced_df['label']

# Initialize TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
print("\nFitting TF-IDF Vectorizer and transforming text data...")

# Fit and transform
x_vectorized = tfidf_vec.fit_transform(xtxt)

import scipy.sparse
x_combined = scipy.sparse.hstack((x_vectorized, xtra_features.values))

# Split into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
    x_combined, ytxt, test_size=0.20, random_state=42, stratify=ytxt
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Training RandomForestClassifier model...")

model.fit(xtrain, ytrain)
print("\n Random Forest Classifier model training complete")

joblib.dump(model, 'spam_detector.pkl')
joblib.dump(tfidf_vec, 'tfidf_vec.pkl')
print("New pkl files saved.")