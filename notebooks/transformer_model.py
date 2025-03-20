# notebooks/transformer_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Load preprocessed data
df = pd.read_csv("C:/Users/Admin/Desktop/python/NLP_PROJECT/data/preprocessed_data.csv")

# Check dataset size
print(f"Dataset size before sampling: {len(df)}")

# Dynamically adjust sample size
sample_size = min(800, len(df))  # Avoid sampling more than available
df = df.sample(n=sample_size, random_state=42)

# Verify unique values in 'Category'
print("Unique values in 'Category':", df["Category"].unique())

# Map string labels to numerical values
label_map = {"trust": 0, "scam": 1}
df["Category"] = df["Category"].map(label_map)

# Verify mapped labels
print("Mapped labels:", df["Category"].unique())

# Check for NaN values
print("NaN values in 'Category':", df["Category"].isna().sum())

# Drop rows with NaN values in 'Category'
df = df.dropna(subset=["Category"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["final_text"], df["Category"], test_size=0.2, random_state=42)

# Reset index for y_train and y_test
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Verify shapes
print("X_train shape:", len(X_train))
print("y_train shape:", len(y_train))
print("X_test shape:", len(X_test))
print("y_test shape:", len(y_test))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-base")
model = AutoModelForSequenceClassification.from_pretrained("Davlan/afro-xlmr-base", num_labels=2)

# Tokenize data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=50)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=50)

# Convert to PyTorch datasets
class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()  # Convert labels to a list

    def __getitem__(self, idx):
        print(f"Index: {idx}, Label: {self.labels[idx]}")  # Debug print
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)  # Ensure labels are integers
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SMSDataset(train_encodings, y_train)
test_dataset = SMSDataset(test_encodings, y_test)

# Define Trainer
training_args = TrainingArguments(
    output_dir="C:/Users/Admin/Desktop/python/NLP_PROJECT/models/afroxlmr",  # Directory to save the model
    eval_strategy="epoch",            # Evaluate after each epoch
    learning_rate=2e-5,               # Learning rate
    per_device_train_batch_size=16,   # Batch size for training
    num_train_epochs=3,               # Number of epochs
    weight_decay=0.01,                # Regularization strength
)

trainer = Trainer(
    model=model,                      # Pre-trained AfroXLMR model
    args=training_args,               # Training configuration
    train_dataset=train_dataset,      # Training data
    eval_dataset=test_dataset,        # Evaluation data
)

# Fine-tune model
trainer.train()

# Save model and tokenizer
model_save_path = "C:/Users/Admin/Desktop/python/NLP_PROJECT/models/afroxlmr"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✅ Transformer model fine-tuning complete! Model and tokenizer saved to '{model_save_path}'.")
