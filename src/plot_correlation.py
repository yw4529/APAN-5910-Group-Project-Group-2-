"""
Generates a scatter plot of sentiment score vs. next-day return,
for inclusion in the report.
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/merged_analysis.csv")
df = df.dropna(subset=["sentiment_score", "return_next_day"])

plt.figure(figsize=(7, 5))
colors = df["sentiment_label"].map({
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
})
plt.scatter(df["sentiment_score"], df["return_next_day"], c=colors, alpha=0.7)
plt.xlabel("Sentiment Confidence Score")
plt.ylabel("Next-Day Return")
plt.title("Headline Sentiment vs. Next-Day Stock Return")
plt.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("reports/sentiment_vs_return.png", dpi=150)
print("Saved plot to reports/sentiment_vs_return.png")
