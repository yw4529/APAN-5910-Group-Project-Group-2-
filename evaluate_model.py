"""
Detailed evaluation of the fine-tuned model on the held-out test split:
classification report + confusion matrix, saved as a PNG for the report/slides.

Usage:
    python modeling/evaluate_model.py \
        --data_dir data/financial_phrasebank \
        --model_dir checkpoints/distilbert-finsent
"""

import argparse

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

LABEL_NAMES = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/financial_phrasebank")
    parser.add_argument("--model_dir", default="checkpoints/distilbert-finsent")
    parser.add_argument("--out_png", default="reports/confusion_matrix.png")
    args, _unknown = parser.parse_known_args()

    dsd = load_from_disk(args.data_dir)
    test = dsd["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text_col = "sentence" if "sentence" in test.column_names else "text"
    texts = test[text_col]
    labels = test["label"]

    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
            enc.pop("token_type_ids", None)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=3))

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()

    import os
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, dpi=150)
    print(f"Confusion matrix saved to {args.out_png}")


if __name__ == "__main__":
    main()
