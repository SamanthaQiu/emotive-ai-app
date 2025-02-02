import pandas as pd
from transformers import pipeline

# Load sentiment analysis pipeline dynamically
sentiment_pipeline = pipeline("sentiment-analysis")

# Load dataset
dataset_path = "IMDB_Dataset.csv"
df = pd.read_csv(dataset_path)

# Apply sentiment analysis with truncation (limit to first 512 characters)
df["predicted_sentiment"] = df["review"].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])

# Convert sentiment labels to 1 (positive) and 0 (negative)
df["predicted_sentiment"] = df["predicted_sentiment"].map({"POSITIVE": 1, "NEGATIVE": 0})

# Save processed file
output_path = "IMDB_Dataset_labeled.csv"
df.to_csv(output_path, index=False)
print(f"Labeled dataset saved as '{output_path}'.")
