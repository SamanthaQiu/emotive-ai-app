import pandas as pd
from textblob import TextBlob  # Import TextBlob for sentiment analysis

# Set the file path of the IMDb dataset (Make sure to update it to your actual path)
dataset_path = "IMDB_Dataset.csv"

# Read the dataset
df = pd.read_csv(dataset_path)

# Function to analyze sentiment of a text
def get_sentiment_score(text):
    analysis = TextBlob(text)  # Create a TextBlob object
    polarity = analysis.sentiment.polarity  # Get sentiment polarity score (-1 to 1)
    
    return 1 if polarity > 0 else 0  # 1 for positive, 0 for negative

# Apply the sentiment function to the dataset
df["sentiment_score"] = df["review"].apply(get_sentiment_score)

# Print the first few rows of the dataset

print(df.head())
# Show dataset information
print(df.info())