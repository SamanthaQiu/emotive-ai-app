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

print(torch.version.cuda)  # 查看 PyTorch 兼容的 CUDA 版本

# 确保 sklearn 已安装，否则安装
try:
    from sklearn.metrics import accuracy_score
except ImportError:
    os.system("pip install scikit-learn")

# 目录路径
output_dir = "imdb-distilbert-finetuned"
log_dir = "logs"

# **检查 logs 是否已经存在**
if os.path.exists(log_dir):
    if not os.path.isdir(log_dir):  
        print(f"⚠️ Error: {log_dir} exists but is not a directory. Removing it...")
        os.remove(log_dir)  # **删除错误的文件**
    else:
        print(f"🗑 Removing existing directory: {log_dir}")
        shutil.rmtree(log_dir)  # **删除整个目录**

# **确保 logs 目录正确创建**
try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"✅ Directory '{log_dir}' successfully created!")
except Exception as e:
    print(f"❌ Failed to create directory '{log_dir}': {e}")
    exit(1)  # **强制退出，避免后续错误**

# 打印 PyTorch 版本信息
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 加载 IMDb 数据集
raw_datasets = load_dataset("imdb")

# 加载预训练的 tokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 定义 tokenization 过程
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# 预处理数据集
encoded_datasets = raw_datasets.map(tokenize_function, batched=True)

# 数据拆分（训练集 90%，验证集 10%）
small_train_dataset = encoded_datasets["train"].train_test_split(test_size=0.1)
train_dataset = small_train_dataset["train"]
eval_dataset = small_train_dataset["test"]
test_dataset = encoded_datasets["test"]

# **💡 修正 train_dataloader 位置**
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# 加载预训练的模型
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=log_dir,  # ✅ 确保 logs 目录正确
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

# 计算准确率的函数
accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  # 未来将被移除
    compute_metrics=compute_metrics
)

# 训练模型
trainer.train()

# 在测试集上评估
metrics = trainer.evaluate(test_dataset)
print("Test set metrics:", metrics)

# 保存训练好的模型
fine_tuned_model_path = "fine_tuned_sentiment_model"
trainer.save_model(fine_tuned_model_path)

# 加载训练好的模型用于推理
sentiment_pipeline = pipeline(
    "text-classification",
    model=fine_tuned_model_path,
    tokenizer=tokenizer,
    return_all_scores=True
)

# 读取 IMDB 数据集进行推理
dataset_path = "IMDB_Dataset.csv"
df = pd.read_csv(dataset_path)

# 使用 fine-tuned 模型进行情感分析
df["predicted_sentiment"] = df["review"].apply(lambda x: sentiment_pipeline(x[:256])[0]['label'])

# 将标签映射为 0（负面）或 1（正面）
df["predicted_sentiment"] = df["predicted_sentiment"].map({"LABEL_1": 1, "LABEL_0": 0})

# 保存结果
output_path = "IMDB_Dataset_labeled.csv"
df.to_csv(output_path, index=False)

print(f"Labeled dataset saved as '{output_path}'.")

# 🎯 添加 Stretch Goal: 让用户输入一个影评，输出情感分析结果
while True:
    user_input = input("输入一条电影评论 (输入 'exit' 退出): ")
    if user_input.lower() == "exit":
        break
    prediction = sentiment_pipeline(user_input)
    print("情感分析结果:", prediction)
