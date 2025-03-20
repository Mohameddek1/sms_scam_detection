from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the saved model and tokenizer
model_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/afroxlmr"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Function to classify text
def classify_message(text):
    """Classifies the input message as 'scam' or 'trust'."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    label_map = {0: "trust", 1: "scam"}
    return label_map[predicted_class]

# Example input text
test_text = "Tuma kwa hii namba"
predicted_label = classify_message(test_text)

# Load test dataset for accuracy calculation
data_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Get predictions for the entire test dataset
true_labels = df["Category"].map({"trust": 0, "scam": 1}).values
predictions = []

for text in df["final_text"]:
    pred_label = classify_message(text)
    predictions.append(0 if pred_label == "trust" else 1)  # Convert back to numeric format

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)

print(f"Input text: {test_text}")
print(f"Predicted label: {predicted_label}")
print(f"Model Accuracy: {accuracy:.2%}")
