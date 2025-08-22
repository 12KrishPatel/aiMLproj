import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

print("First 5 rows of 'text_content' and 'label':")
print(combined_df[['text', 'label']].head())

print("\nDistribution of 'label' in new dataset:")
print(combined_df['label'].value_counts())
print(combined_df['label'].value_counts(normalize=True))

# Seperate X (features) and Y (target)
xtxt = combined_df['text']
ytxt = combined_df['label']

# Initialize TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
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


# ADDING A NEW DATASET TO TEST AGAINST

# Load the new dataset
try:
    emails_df = pd.read_csv('emails.csv')
except FileNotFoundError:
    print("Emails.csv not found")
    exit()
except Exception as e:
    print("Error while loading emails.csv")
    exit()

# ID text and spam cols in emails_df
xemail = emails_df['text']
yemail = emails_df['spam']

# Preprocess and vectorize new data
print("\n Transforming text data using already fitted TD-IDF Vector")
xemail_vec = tfidf_vec.transform(xemail)
print("New email data vectorization complete!")

# Make predictions
print("\nMaking predictions on the 'emails.csv' dataset using the trained model...")
yemail_pred = model_text.predict(xemail_vec)
print("Predictions complete!")
print("\nFirst 10 actual labels from 'emails.csv' (y_email):")
print(yemail.head(10).tolist()) 
print("\nFirst 10 predicted labels for 'emails.csv' (y_email_pred):")
print(yemail_pred[:10].tolist()) 

# Evaluate Performance
print("\n--- Model Evaluation on 'emails.csv' ---")

accuracy_email = accuracy_score(yemail, yemail_pred)
print(f"Accuracy ('emails.csv'): {accuracy_email:.4f}")

precision_email = precision_score(yemail, yemail_pred)
print(f"Precision ('emails.csv'): {precision_email:.4f}")

recall_email = recall_score(yemail, yemail_pred)
print(f"Recall ('emails.csv'): {recall_email:.4f}")

f1_email = f1_score(yemail, yemail_pred)
print(f"F1-Score ('emails.csv'): {f1_email:.4f}")

print("\nConfusion Matrix ('emails.csv'):")
print(confusion_matrix(yemail, yemail_pred))


print("\nClassification Report ('emails.csv'):")
print(classification_report(yemail, yemail_pred))


# Saving the trained model
joblib.dump(model_text, 'spam_detector.pkl')
print("Trained model saved as spam_detector.pkl")

# Save th TF-IDF vec
joblib.dump(tfidf_vec, 'tfidf_vec.pkl')
print("Tfidf saved as tfidf_vec.pkl")

# Spam detection function
def spam_predictor(input):
    try: 
        # Load the two components
        loaded_vec = joblib.load('tfidf_vec.pkl')
        loaded_model = joblib.load('spam_detector.pkl')

        # Transform input text with vectorizer
        text_series = pd.Series([input])
        text_vec = loaded_vec.transform(text_series)

        # Make prediction
        prediction = loaded_model.predict(text_vec)[0]

        if prediction == 1:
            return "SPAM"
        else:
            return "NOT SPAM"
    except FileNotFoundError:
        return "Model or vectorizer files not found"
    except Exception as e:
        return "Error occured during prediction"