# APAN 5910 Final Project: Financial Sentiment Predictor

A fine-tuned DistilBERT model for financial headline sentiment classification,
with analysis of sentiment vs. next-day stock returns and a live Streamlit demo.

## Team
- Jialu Huang — modeling, data preparation, fine-tuning, evaluation
- Yu Wu (Olivia) — price data, correlation analysis, web interface

## Model performance
- Test accuracy: 97.3%
- F1 macro: 0.962
- Fine-tuned on Financial PhraseBank (Malo et al., 2014)

## Setup

```bash
uv sync
```

Then place the trained model checkpoint files in `checkpoints/distilbert-finsent/`
(not included in this repo due to file size; contact the team for the checkpoint).

## Usage

```bash
# Fetch historical price data
uv run python src/fetch_prices.py

# Test the sentiment model
uv run python inference.py

# Run sentiment scoring + correlation analysis
uv run python src/evaluate.py

# Generate correlation plot
uv run python src/plot_correlation.py

# Launch the web app
uv run streamlit run app.py
```

## Data sources
- Financial PhraseBank (Malo et al., 2014): https://huggingface.co/datasets/FinanceMTEB/financial_phrasebank
- Historical stock prices via yfinance: https://pypi.org/project/yfinance/

## Notes
- Environment setup details for Intel Mac compatibility are in `SETUP_NOTES.md`.
- Tickers analyzed: AAPL, MSFT, TSLA, NVDA.
