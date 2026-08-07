"""
Clean, reusable inference interface for the fine-tuned sentiment model.

This is the hand-off point to Person B's app/analysis code — it should
be the ONLY file they need to import from your half of the project.

Usage from the Streamlit app:

    from modeling.inference import SentimentPredictor

    predictor = SentimentPredictor("checkpoints/distilbert-finsent")
    result = predictor.predict("Apple beats earnings expectations for Q3")
    # result = {"label": "positive", "score": 0.94,
    #           "probs": {"negative": 0.02, "neutral": 0.04, "positive": 0.94}}
"""

from dataclasses import dataclass
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class SentimentResult:
    label: str
    score: float  # confidence of the predicted label
    probs: dict   # full probability distribution over all labels

    def to_dict(self) -> dict:
        return {"label": self.label, "score": self.score, "probs": self.probs}


class SentimentPredictor:
    """Loads the fine-tuned model once and exposes a simple predict() call."""

    def __init__(self, model_dir: str = "checkpoints/distilbert-finsent", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str) -> SentimentResult:
        enc = self.tokenizer(text, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        enc.pop("token_type_ids", None)
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        pred_id = int(probs.argmax())
        label = self.id2label[pred_id]
        prob_dict = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return SentimentResult(label=label, score=float(probs[pred_id]), probs=prob_dict)

    @torch.no_grad()
    def predict_batch(self, texts: list) -> list:
        """Batched version for scoring many headlines at once (e.g. for the
        return-correlation analysis Person B runs)."""
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.device)
        enc.pop("token_type_ids", None)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for row in probs:
            pred_id = int(row.argmax())
            label = self.id2label[pred_id]
            prob_dict = {self.id2label[i]: float(row[i]) for i in range(len(row))}
            results.append(SentimentResult(label=label, score=float(row[pred_id]), probs=prob_dict))
        return results


# Convenience module-level function if a global singleton is preferred
_default_predictor: Union[SentimentPredictor, None] = None


def predict_sentiment(text: str, model_dir: str = "checkpoints/distilbert-finsent") -> dict:
    """Simplest possible entry point: predict_sentiment("some headline") -> dict"""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = SentimentPredictor(model_dir)
    return _default_predictor.predict(text).to_dict()


if __name__ == "__main__":
    # Quick manual smoke test
    predictor = SentimentPredictor()
    examples = [
        "Company reports record profits, beating analyst expectations",
        "Shares plunge after disappointing guidance for next quarter",
        "The board announced no changes to its dividend policy",
    ]
    for ex in examples:
        result = predictor.predict(ex)
        print(f"{ex}\n  -> {result.label} ({result.score:.3f})\n")
