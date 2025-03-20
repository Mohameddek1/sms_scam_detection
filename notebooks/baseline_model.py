# notebooks/baseline_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Load preprocessed data
df = pd.read_csv("C:/Users/Admin/Desktop/python/NLP_PROJECT/data/preprocessed_data.csv")

# Feature engineering
# Converts text into a numerical format that a machine learning model can understand.
vectorizer = TfidfVectorizer(max_features=5000)   # Keeps only the top 5000 most important words from the text dataset.
X = vectorizer.fit_transform(df["final_text"])
y = df["Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/baseline_model.pkl")
joblib.dump(vectorizer, "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/tfidf_vectorizer.pkl")
print("✅ Baseline model training complete! Model saved to '../models/baseline_model.pkl'.")

# Generate Word Clouds for Scam and Non-Scam Messages
output_dir = "C:/Users/Admin/Desktop/python/NLP_PROJECT/reports/visualizations"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Word Cloud for Non-Scam Messages
non_scam_text = " ".join(df[df["Category"] == "trust"]["final_text"])
non_scam_wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color="white",  # Background color
    max_words=200,            # Maximum number of words to display
    colormap="viridis"        # Color scheme
).generate(non_scam_text)

# Plot and save non-scam word cloud
plt.figure(figsize=(10, 5))
plt.imshow(non_scam_wordcloud, interpolation="bilinear")
plt.axis("off")  # Remove axes
plt.title("Word Cloud for Non-Scam Messages", fontsize=16)
plt.show()
non_scam_wordcloud.to_file(os.path.join(output_dir, "non_scam_wordcloud.png"))
print(f"✅ Word cloud for non-scam messages saved to '{output_dir}/non_scam_wordcloud.png'.")

# Word Cloud for Scam Messages
scam_text = " ".join(df[df["Category"] == "scam"]["final_text"])
scam_wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color="white",  # Background color
    max_words=200,            # Maximum number of words to display
    colormap="plasma"         # Color scheme
).generate(scam_text)

# Plot and save scam word cloud
plt.figure(figsize=(10, 5))
plt.imshow(scam_wordcloud, interpolation="bilinear")
plt.axis("off")  # Remove axes
plt.title("Word Cloud for Scam Messages", fontsize=16)
plt.show()
scam_wordcloud.to_file(os.path.join(output_dir, "scam_wordcloud.png"))
print(f"✅ Word cloud for scam messages saved to '{output_dir}/scam_wordcloud.png'.")