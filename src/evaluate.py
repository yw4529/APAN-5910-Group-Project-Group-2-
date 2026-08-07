"""
Scores collected headlines with the fine-tuned sentiment model, then merges
with next-day stock returns to test for correlation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from inference import SentimentPredictor
import pandas as pd


def score_headlines():
    predictor = SentimentPredictor("checkpoints/distilbert-finsent")
    headlines = pd.read_csv("data/headlines.csv")
    results = predictor.predict_batch(headlines["headline"].tolist())
    headlines["sentiment_label"] = [r.label for r in results]
    headlines["sentiment_score"] = [r.score for r in results]
    headlines.to_csv("data/headlines_scored.csv", index=False)
    print(headlines[["ticker", "headline", "sentiment_label", "sentiment_score"]])
    return headlines


def merge_with_returns(headlines):
    merged_rows = []
    for ticker in headlines["ticker"].unique():
        price = pd.read_csv(f"data/prices/{ticker}.csv")
        price["Date"] = pd.to_datetime(price.iloc[:, 0], errors="coerce")
        price = price.dropna(subset=["Date"])
        price["return_next_day"] = price["Close"].astype(float).pct_change().shift(-1)

        sub = headlines[headlines["ticker"] == ticker].copy()
        sub["date"] = pd.to_datetime(sub["date"])

        merged = sub.merge(
            price[["Date", "return_next_day"]],
            left_on="date", right_on="Date", how="left"
        )
        merged_rows.append(merged)
    return pd.concat(merged_rows)


if __name__ == "__main__":
    headlines = score_headlines()
    merged = merge_with_returns(headlines)
    merged.to_csv("data/merged_analysis.csv", index=False)
    print("\nMerged data:")
    print(merged[["ticker", "date", "sentiment_score", "return_next_day"]])
    print("\nCorrelation:")
    print(merged[["sentiment_score", "return_next_day"]].corr())
