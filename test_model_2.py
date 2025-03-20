import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the saved model and vectorizer
model_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/baseline_model.pkl"
vectorizer_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/tfidf_vectorizer.pkl"
data_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/data/preprocessed_data.csv"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def classify_message(text):
    """Classifies the input message as 'scam' or 'trust'."""
    transformed_text = vectorizer.transform([text])  # Convert text into numerical format
    prediction = model.predict(transformed_text)[0]  # Predict category
    return prediction

# Example input text
test_text = "Niaje"

# Classify the message
predicted_label = classify_message(test_text)

# Load test data to calculate accuracy
df = pd.read_csv(data_path)
X_test = vectorizer.transform(df["final_text"])  # Convert all text into numerical format
y_test = df["Category"]
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Input text: {test_text}")
print(f"Predicted label: {predicted_label}")
print(f"Model Accuracy: {accuracy:.2%}")
