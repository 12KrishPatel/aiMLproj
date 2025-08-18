import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Esnsure messages DF loads
try: 
    messages_df = pd.read_csv('messages.csv')
except FileNotFoundError:
    print("Error: messages.csv not found")
    exit()

# Fill w/ empty strings to avoid errors
messages_df['subject'] = messages_df['subject'].fillna('')
messages_df['message'] = messages_df['message'].fillna('')

# Combine messages and subjects into one column
messages_df['text'] = messages_df['subject'] + "" + messages_df['message']

print("First 5 rows of 'text_content' and 'label':")
print(messages_df[['text', 'label']].head())

print("\nDistribution of 'label' in new dataset:")
print(messages_df['label'].value_counts())
print(messages_df['label'].value_counts(normalize=True))

# Seperate X (features) and Y (target)
xtxt = messages_df['text']
ytxt = messages_df['label']

# Initialize TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words='english')
print("\nFitting TF-IDF Vectorizer and transforming text data...")

# Fit and transform
x_vectorized = tfidf_vec.fit_transform(xtxt)

# Split into training and testing sets
xtrain_text, xtest_text, ytrain_text, ytest_text = train_test_split(
    x_vectorized, ytxt, test_size=0.20, random_state=42, stratify=ytxt
)

# Train the model with Logistic Regression
model_text = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
print(f"\nTraining {type(model_text).__name__} model on text features...")

model_text.fit(xtrain_text, ytrain_text)
print(f"{type(model_text).__name__} model training complete!")

y_pred_text = model_text.predict(xtest_text)

print("\n--- Predictions on New Model ---")
print("First 10 actual labels (y_test_text):")
print(ytest_text.head(10).tolist())
print("\nFirst 10 predicted labels (y_pred_text):")
print(y_pred_text[:10].tolist())
