import torch
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

# **检查 PyTorch 和 CUDA 版本**
def check_env():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

# **加载 IMDb 数据集**
def load_and_preprocess_data():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    # **减少数据量，提高训练速度**
    dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    return train_dataset, eval_dataset, tokenizer

# **训练模型**
def train_model(train_dataset, eval_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="finetuned_sentiment_model",
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

    accuracy_metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("finetuned_sentiment_model")
    return model, tokenizer

# **加载已训练模型**
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("finetuned_sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# **情感分析**
def get_sentiment_label(sentiment_pipeline, text):
    result = sentiment_pipeline(text[:256])[0]
    label = result["label"]
    score = result["score"]

    # **调整负面情绪评分为负数**
    if label == "LABEL_0":  # Negative
        score = -score

    sentiment = "POSITIVE" if score > 0 else "NEGATIVE"
    return sentiment, score

# **主函数**
def main():
    check_env()
    train_dataset, eval_dataset, tokenizer = load_and_preprocess_data()
    model, tokenizer = train_model(train_dataset, eval_dataset, tokenizer)
    sentiment_pipeline = load_model()

    # **交互模式**
    while True:
        user_input = input("Enter a review (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        sentiment, score = get_sentiment_label(sentiment_pipeline, user_input)
        print(f"Sentiment Analysis Result: {sentiment} (Score: {score:.2f}, Range: -1 to 1)")

if __name__ == "__main__":
    main()
