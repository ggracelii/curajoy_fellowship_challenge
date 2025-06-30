# Cyberbullying Detection with BERT

# Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import re
import requests
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from datasets import Dataset, DatasetDict, Value, Features
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EvalPrediction
)

from transformers.utils import logging
logging.set_verbosity_error()

# Load and parse dataset
url = "https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/refs/heads/master/data/dataset.txt"
lines = requests.get(url).text.splitlines()

records = []
for line in lines:
    m = re.search(r':\s*(True|False)[,\}]?$', line.strip())
    if not m:
        continue
    label = m.group(1) == 'True'
    text = line[:m.start()].strip().strip('"\'')
    records.append((text, label))

df = pd.DataFrame(records, columns=['text', 'label'])

# Balance the dataset
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]
df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

# Tokenization with DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True, max_length=128)

# Prepare HuggingFace dataset
train_df, test_df = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42)

train_df = train_df[['text', 'label']].copy()
test_df = test_df[['text', 'label']].copy()
train_df['label'] = train_df['label'].astype(np.float32)
test_df['label'] = test_df['label'].astype(np.float32)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

features = Features({'text': Value('string'), 'label': Value('int64')})
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df, features=features),
    "test": Dataset.from_pandas(test_df, features=features),
})
tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds = tokenized_ds.remove_columns(['text'])
tokenized_ds.set_format('torch')

# Load BERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Evaluation metrics
def compute_metrics(pred: EvalPrediction):
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
    y_true = pred.label_ids

    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    y_pred = (probs > best_thresh).astype(int)

    print(f"\nBERT optimal threshold: {best_thresh:.3f} (P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, F1={f1s[best_idx]:.3f})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"\nAcc: {accuracy_score(y_true, y_pred):.3f}, "
          f"Prec: {precision_score(y_true, y_pred, zero_division=0):.3f}, "
          f"Rec: {recall_score(y_true, y_pred, zero_division=0):.3f}, "
          f"F1: {f1_score(y_true, y_pred):.3f}")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }

# Define training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()