"""
Fine-tunes DistilBERT (or another base checkpoint, e.g. FinBERT) for
3-class financial sentiment classification on Financial PhraseBank.

Usage:
    python modeling/train.py \
        --data_dir data/financial_phrasebank \
        --model_name distilbert-base-uncased \
        --output_dir checkpoints/distilbert-finsent \
        --epochs 4
"""

import argparse

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

LABEL_NAMES = ["negative", "neutral", "positive"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def build_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/financial_phrasebank")
    parser.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        help="Base checkpoint. Try 'ProsusAI/finbert' as an alternative base.",
    )
    parser.add_argument("--output_dir", default="checkpoints/distilbert-finsent")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=64)
    args, _unknown = parser.parse_known_args()

    dsd = load_from_disk(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"] if "sentence" in batch else batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dsd.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()

    # Save the final (best) model + tokenizer for downstream inference
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")

    # Quick sanity check on the held-out test set
    test_metrics = trainer.evaluate(tokenized["test"])
    print("Test set metrics:", test_metrics)


if __name__ == "__main__":
    main()