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

print(torch.version.cuda)  # Check the CUDA version compatible with PyTorch

# Ensure sklearn is installed; otherwise, install it
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    os.system("pip install scikit-learn")

# Directory paths
output_dir = "imdb-distilbert-finetuned"
log_dir = "logs"

# **Check if logs directory already exists**
if os.path.exists(log_dir):
    if not os.path.isdir(log_dir):  
        print(f"⚠️ Error: {log_dir} exists but is not a directory. Removing it...")
        os.remove(log_dir)  # **Remove incorrect file**
    else:
        print(f"🗑 Removing existing directory: {log_dir}")
        shutil.rmtree(log_dir)  # **Remove entire directory**

# **Ensure logs directory is properly created**
try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"✅ Directory '{log_dir}' successfully created!")
except Exception as e:
    print(f"❌ Failed to create directory '{log_dir}': {e}")
    exit(1)  # **Force exit to prevent further errors**

# Print PyTorch version information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load IMDb dataset
raw_datasets = load_dataset("imdb")

# Load pre-trained tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Define tokenization process
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Preprocess dataset
encoded_datasets = raw_datasets.map(tokenize_function, batched=True)

# Split dataset (90% training, 10% validation)
small_train_dataset = encoded_datasets["train"].train_test_split(test_size=0.1)
train_dataset = small_train_dataset["train"]
eval_dataset = small_train_dataset["test"]
test_dataset = encoded_datasets["test"]

# **Fix train_dataloader location**
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Load pre-trained model
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Training configuration
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=log_dir,  # ✅ Ensure logs directory is correct
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

# Function to compute accuracy
accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  # Will be deprecated in future versions
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate on the test set
metrics = trainer.evaluate(test_dataset)
print("Test set metrics:", metrics)

# Save fine-tuned model
fine_tuned_model_path = "fine_tuned_sentiment_model"
trainer.save_model(fine_tuned_model_path)

# Load trained model for inference
sentiment_pipeline = pipeline(
    "text-classification",
    model=fine_tuned_model_path,
    tokenizer=tokenizer,
    top_k=1  # Replace return_all_scores=True to avoid format errors
)

# Load IMDb dataset for inference
dataset_path = "IMDB_Dataset.csv"
df = pd.read_csv(dataset_path)

# **Fix sentiment_pipeline return format**
def get_sentiment_label(text):
    try:
        result = sentiment_pipeline(text[:256])  # Run sentiment analysis
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0:
            return result[0][0].get('label', "UNKNOWN")  # Extract label from nested list
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            return result[0].get('label', "UNKNOWN")  # Support previous format
        else:
            print(f"Unexpected result format: {result}")
            return "UNKNOWN"
    except Exception as e:
        print(f"Error processing text: {text[:50]}... - {e}")
        return "ERROR"

df["predicted_sentiment"] = df["review"].apply(get_sentiment_label)

# Map labels to 0 (negative) or 1 (positive)
df["predicted_sentiment"] = df["predicted_sentiment"].map({"LABEL_1": 1, "LABEL_0": 0}).fillna(-1)  # Handle unknown cases

# Save results
output_path = "IMDB_Dataset_labeled.csv"
df.to_csv(output_path, index=False)

print(f"Labeled dataset saved as '{output_path}'.")

# Interactive sentiment analysis
while True:
    user_input = input("Enter a movie review (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    prediction = get_sentiment_label(user_input)
    print("Sentiment analysis result:", prediction)
