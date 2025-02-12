import pandas as pd
from datasets import load_dataset
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
from torch.utils.data import DataLoader

# Check PyTorch and CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

def get_sentiment_label(sentiment_pipeline, text):
    """
    Predicts the sentiment label of a given text using a trained sentiment analysis model.
    """
    try:
        result = sentiment_pipeline(text[:256])

        # Handle different output formats from the sentiment analysis pipeline
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0:
            result = result[0][0]
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            result = result[0]
        else:
            print(f"Unexpected result format: {result}")
            return "ERROR"

        label = result["label"]
        score = result["score"]

        # Define the logic for positive and negative classification
        if label == "LABEL_1" or (label == "LABEL_0" and score < 0.5):  
            return "Positive", score
        else:
            return "Negative", score

    except Exception as e:
        print(f"Error processing text: {text[:50]}... - {e}")
        return "ERROR", 0.0

def tokenize_function(examples, tokenizer):
    """
    Tokenizes the input text dataset.
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

def compute_metrics(eval_preds, accuracy_metric):
    """
    Computes accuracy metrics for model evaluation.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def train_model(train_dataset, eval_dataset, tokenizer):
    """
    Trains the DistilBERT model for sentiment analysis and returns the trained model.
    """
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="imdb-distilbert-finetuned",
        logging_dir="logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True
    )

    accuracy_metric = load_metric("accuracy")  # Define accuracy metric inside function

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, accuracy_metric)  # Pass metric
    )

    trainer.train()
    return model, tokenizer, trainer

def setup_environment():
    """
    Sets up directories and loads the dataset.
    """
    log_dir = "logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Load the IMDb dataset
    raw_datasets = load_dataset("imdb")

    # Reduce dataset size for faster training
    train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = raw_datasets["test"].shuffle(seed=42).select(range(500))

    return train_dataset, eval_dataset

if __name__ == "__main__":
    # Set up environment and load data
    train_dataset, eval_dataset = setup_environment()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenize datasets
    encoded_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    encoded_eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Train model
    model, tokenizer, trainer = train_model(encoded_train_dataset, encoded_eval_dataset, tokenizer)

    # Save the trained model
    fine_tuned_model_path = "fine_tuned_sentiment_model"
    trainer.save_model(fine_tuned_model_path)

    # Load trained model for inference
    sentiment_pipeline = pipeline(
        "text-classification",
        model=fine_tuned_model_path,
        tokenizer=tokenizer,
        top_k=1
    )

    # Interactive sentiment analysis with scores
    while True:
        user_input = input("Enter a movie review (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        sentiment, confidence = get_sentiment_label(sentiment_pipeline, user_input)
        print(f"Sentiment Analysis Result: {sentiment} (Confidence: {confidence:.2f})")
