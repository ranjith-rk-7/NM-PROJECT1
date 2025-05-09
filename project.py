

# Step 2: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 3: Load and Label Data (skip bad rows)
fake_df = pd.read_csv("Fake.csv", engine="python", on_bad_lines='skip')
true_df = pd.read_csv("True.csv", engine="python", on_bad_lines='skip')


fake_df["label"] = 0  # fake news
true_df["label"] = 1  # real news

# Step 4: Combine and Shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
X = df["text"]
y = df["label"]

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Step 9: User Input for Prediction
print("\n--- News Prediction ---")
user_input = input("Enter a news article to check if it's real or fake:\n\n")
input_tfidf = vectorizer.transform([user_input])
prediction = model.predict(input_tfidf)[0]

print("\nPrediction: This news is", "🟢 Real" if prediction == 1 else "🔴 Fake")

