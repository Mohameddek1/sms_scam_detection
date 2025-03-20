# notebooks/preprocessing.py
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("C:/Users/Admin/Desktop/python/NLP_PROJECT/data/bongo_scam.csv")

# Handle missing values
df["Sms"] = df["Sms"].fillna("")

# Remove duplicates
df = df.drop_duplicates()

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["cleaned_text"] = df["Sms"].apply(clean_text)

# Tokenization
df["tokens"] = df["cleaned_text"].apply(lambda text: word_tokenize(text, preserve_line=True))

# Stopword removal
swahili_stopwords = {
    "akasema", "alikuwa", "alisema", "baada", "basi", "bila", "cha", "chini", "hadi", "hapo", "hata", "hivyo", "hiyo",
    "huku", "huo", "ili", "ilikuwa", "juu", "kama", "karibu", "katika", "kila", "kima", "kisha", "kubwa", "kutoka",
    "kuwa", "kwa", "kwamba", "kwenda", "kwenye", "la", "lakini", "mara", "mdogo", "mimi", "mkubwa", "mmoja", "moja",
    "muda", "mwenye", "na", "naye", "ndani", "ng", "ni", "nini", "pamoja", "pia", "sana", "sasa", "sauti", "tafadhali",
    "tena", "tu", "vile", "wa", "wakati", "wake", "walikuwa", "wao", "watu", "wengine", "wote", "ya", "yake", "yangu",
    "yao", "yeye", "yule", "za", "zao", "zile"
}

try:
    swahili_stopwords = set(stopwords.words("swahili"))
except (LookupError, OSError):
    print("Swahili stopwords not found in NLTK. Using custom Swahili stopwords list.")

df["filtered_tokens"] = df["tokens"].apply(lambda words: [word for word in words if word not in swahili_stopwords])

# Convert tokens back to text
df["final_text"] = df["filtered_tokens"].apply(lambda words: " ".join(words))

# Save preprocessed data
df.to_csv("C:/Users/Admin/Desktop/python/NLP_PROJECT/data/preprocessed_data.csv", index=False)
print("✅ Preprocessing complete! Data saved to '../data/preprocessed_data.csv'.")