import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import numpy as np
from evaluate import load as load_metric
import torch
import os
import shutil

# Check PyTorch and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Function to tokenize input texts
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["review"], truncation=True, padding="max_length", max_length=256)

# Compute accuracy for evaluation
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = load_metric("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Preprocess Amazon dataset
def preprocess_amazon_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"Available columns in Amazon dataset: {df.columns}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=["review", "sentiment"])

    # Convert sentiments to binary: Positive (1) and Negative (0)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x.lower() == "positive" else 0)

    dataset = Dataset.from_pandas(df)
    return dataset

# Load and preprocess IMDb dataset
def load_imdb_dataset():
    imdb_data = load_dataset("imdb")
    train_dataset = imdb_data["train"].shuffle(seed=42).select(range(500))  # Reduce for faster training
    eval_dataset = imdb_data["test"].shuffle(seed=42).select(range(250))

    # Rename 'text' column to 'review' to match Amazon dataset
    train_dataset = train_dataset.rename_columns({"text": "review"})
    eval_dataset = eval_dataset.rename_columns({"text": "review"})

    print("IMDb dataset loaded successfully.")
    return train_dataset, eval_dataset

# Train the model
def train_model(train_dataset, eval_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="fine_tuned_sentiment_model",
        logging_dir="logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,  # Reduced for faster training
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return model, tokenizer, trainer

# Set up environment and load datasets
def setup_environment():
    # Load IMDb dataset
    train_dataset, eval_dataset = load_imdb_dataset()

    # Load Amazon dataset
    amazon_dataset = preprocess_amazon_dataset("amazon_sales_dataset.csv")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize datasets
    encoded_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    encoded_amazon_dataset = amazon_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    return encoded_train_dataset, encoded_amazon_dataset, tokenizer

# Inference function for user input
def analyze_sentiment(sentiment_pipeline, text):
    try:
        result = sentiment_pipeline(text[:256])  # Truncate input for consistency
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            label = result[0]['label']
            score = result[0]['score']
            sentiment = "Positive" if "POSITIVE" in label.upper() else "Negative"
            return sentiment, score
        else:
            return "ERROR", 0.0
    except Exception as e:
        print(f"Error processing input: {e}")
        return "ERROR", 0.0

# Main execution block
if __name__ == "__main__":
    imdb_train, amazon_train, tokenizer = setup_environment()

    # Train on IMDb dataset
    print("Training on IMDb dataset...")
    model, tokenizer, trainer = train_model(imdb_train, imdb_train, tokenizer)

    # Fine-tune on Amazon dataset
    print("Fine-tuning on Amazon dataset...")
    model, tokenizer, trainer = train_model(amazon_train, amazon_train, tokenizer)

    # Save the final fine-tuned model
    model_path = "final_fine_tuned_model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_path)

    print("Model trained and saved successfully!")

    # Load trained model for predictions
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

    # Interactive sentiment analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")

    while True:
        user_input = input("Enter a review (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        sentiment, confidence = analyze_sentiment(sentiment_pipeline, user_input)
        print(f"Sentiment Analysis Result: {sentiment} (Confidence: {confidence:.2f})")
