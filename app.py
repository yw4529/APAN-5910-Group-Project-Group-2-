import streamlit as st
import pandas as pd
from inference import SentimentPredictor

st.set_page_config(
    page_title="Financial Sentiment Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    font-weight: 800;
    letter-spacing: -0.5px;
}
div[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 20px;
}
</style>
""", unsafe_allow_html=True)

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#94a3b8",
}


@st.cache_resource
def load_model():
    return SentimentPredictor("checkpoints/distilbert-finsent")


@st.cache_data
def load_price_data(ticker):
    df = pd.read_csv(
        f"data/prices/{ticker}.csv",
        skiprows=3,
        header=None,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["Date"],
        index_col="Date",
    )
    return df


predictor = load_model()

st.title("📈 Financial Sentiment Predictor")
st.caption("Fine-tuned DistilBERT model for financial headline sentiment classification")

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("Predict Sentiment")
    text = st.text_area(
        "Enter a financial headline:",
        "Apple beats earnings expectations for Q3",
        height=100,
    )
    predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        result = predictor.predict(text)
        color = SENTIMENT_COLORS.get(result.label, "#94a3b8")

        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 12px; background-color: {color}22;
                        border: 1px solid {color}55; margin-top: 10px;">
                <div style="font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
                    Predicted Sentiment
                </div>
                <div style="font-size: 36px; font-weight: 800; color: {color}; text-transform: capitalize;">
                    {result.label}
                </div>
                <div style="font-size: 14px; color: #94a3b8; margin-top: 8px;">
                    Confidence: {result.score:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        prob_df = pd.DataFrame({
            "Sentiment": list(result.probs.keys()),
            "Probability": list(result.probs.values()),
        })
        st.bar_chart(prob_df.set_index("Sentiment"), color=color, height=200)

with col_right:
    st.subheader("Stock Price Trend")
    ticker = st.selectbox("Select a ticker:", TICKERS)

    try:
        price_df = load_price_data(ticker)
        close_col = "Close" if "Close" in price_df.columns else price_df.columns[0]
        chart_df = price_df[[close_col]].copy()
        st.line_chart(chart_df, height=280, color="#3b82f6", use_container_width=True)

        latest_price = price_df[close_col].iloc[-1]
        prev_price = price_df[close_col].iloc[-2]
        pct_change = (latest_price - prev_price) / prev_price * 100

        m1, m2 = st.columns(2)
        m1.metric("Latest Close", f"${latest_price:,.2f}")
        m2.metric("1-Day Change", f"{pct_change:+.2f}%")
    except FileNotFoundError:
        st.info(f"Price data for {ticker} not found. Run `src/fetch_prices.py` first.")

st.divider()
st.caption(
    "Model: fine-tuned DistilBERT on Financial PhraseBank · "
    "Test accuracy 97.3% · F1 macro 0.962"
)
