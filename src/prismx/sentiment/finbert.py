"""FinBERT scoring with an explicit optional dependency boundary."""

from __future__ import annotations

import pandas as pd


def score_finbert(
    news: pd.DataFrame,
    text_column: str = "title",
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Score financial-news text with FinBERT.

    The first invocation downloads model weights through Transformers. Results
    must be timestamp-aligned before they are used in predictive modelling.
    """

    if text_column not in news:
        raise ValueError(f"News is missing text column: {text_column}")
    try:
        from transformers import pipeline
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "Install FinBERT dependencies with: "
            "python -m pip install -e \".[sentiment]\""
        ) from error

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        # Force the PyTorch backend.  Streamlit Cloud currently installs
        # TensorFlow for the LSTM, and Transformers otherwise tries to use
        # TensorFlow first; that path is incompatible with Keras 3.
        framework="pt",
        truncation=True,
    )
    texts = news[text_column].astype(str).tolist()
    predictions = classifier(texts, batch_size=batch_size)
    scored = news.copy()
    scored["sentiment_label"] = [prediction["label"].lower() for prediction in predictions]
    scored["sentiment_score"] = [float(prediction["score"]) for prediction in predictions]
    scored["sentiment_signal"] = scored["sentiment_label"].map(
        {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    ) * scored["sentiment_score"]
    return scored


def aggregate_sentiment(
    scored_news: pd.DataFrame,
    frequency: str = "D",
) -> pd.DataFrame:
    """Aggregate timestamped sentiment into daily features."""

    required = {"published_at", "sentiment_signal"}
    missing = required.difference(scored_news.columns)
    if missing:
        raise ValueError(f"Scored news is missing columns: {', '.join(sorted(missing))}")
    data = scored_news.copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    data["date"] = data["published_at"].dt.tz_convert(None).dt.floor(frequency)
    grouping = ["date"] + (["ticker"] if "ticker" in data else [])
    return data.groupby(grouping).agg(
        sentiment_mean=("sentiment_signal", "mean"),
        sentiment_std=("sentiment_signal", "std"),
        news_count=("sentiment_signal", "count"),
    ).reset_index()
