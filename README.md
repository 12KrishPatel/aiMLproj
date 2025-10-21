📧 Spam Email Detector

This project is a spam and phishing email detector that uses machine learning to tell if a message is spam or not.
It has a simple Streamlit web app where you can paste an email message and see the result instantly.

💻 Link to published website:

spamdetectorkp.streamlit.app

🧠 What It Does

Reads email text and checks if it’s spam

Uses a trained model to predict results

Shows how confident the model is

You can retrain the model with your own data if you want

🗂️ Files in This Project
app.py                → Streamlit web app
messagesmodel.py      → Trains the spam detection model
messages.csv          → Regular email dataset
phishing_email.csv    → Spam/phishing dataset
spam_detector.pkl     → Saved model
tfidf_vec.pkl         → Saved text vectorizer
requirements.txt      → Needed Python packages

⚙️ How to Run It

Clone the project

-git clone https://github.com/yourusername/spam-email-detector.git

-cd spam-email-detector

Install the requirements

-pip install -r requirements.txt

Start the app

-streamlit run app.py

Go to the link Streamlit gives you (usually http://localhost:8501), type or paste an email, and hit Predict.

🧩 If You Want to Retrain the Model

Run this command:

python messagesmodel.py

That will rebuild the spam detector using the CSV files.

🧾 Example
Message	Result	Confidence
"Your account has been suspended, click here to verify."	🛑 SPAM	97%
"Let’s meet tomorrow for the project."	✅ NOT SPAM	95%

👨‍💻 Made by
Krish Patel
(Personal project on spam detection using Python and Streamlit)
