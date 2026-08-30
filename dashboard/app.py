from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Apple Market Movement Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .project-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .signal-up {
        background: #dcfce7;
        color: #166534;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #16a34a;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .signal-down {
        background: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 12px;
        border-left: 6px solid #dc2626;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .information-box {
        background: #eff6ff;
        color: #1e3a8a;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------

@st.cache_data
def load_csv_file(file_path, date_columns=None):
    dataframe = pd.read_csv(file_path)

    if date_columns:
        for column in date_columns:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(
                    dataframe[column],
                    errors="coerce"
                )

    return dataframe


@st.cache_resource
def load_project_models():
    movement_bundle = joblib.load(
        MODELS_DIR
        / "aapl_movement_logistic_model.joblib"
    )

    sentiment_pipeline = joblib.load(
        MODELS_DIR
        / "financial_sentiment_pipeline.joblib"
    )

    return movement_bundle, sentiment_pipeline


def load_optional_csv(file_path, date_columns=None):
    if not file_path.exists():
        return pd.DataFrame()

    return load_csv_file(
        str(file_path),
        date_columns
    )


try:
    market_data = load_csv_file(
        str(
            PROCESSED_DATA_DIR
            / "AAPL_market_features.csv"
        ),
        ["Date"]
    )

    modeling_history = load_csv_file(
        str(
            PROCESSED_DATA_DIR
            / "AAPL_modeling_dataset.csv"
        ),
        ["Date"]
    )

    daily_sentiment = load_csv_file(
        str(
            PROCESSED_DATA_DIR
            / "AAPL_daily_sentiment.csv"
        ),
        ["Trading_Date"]
    )

    test_results = load_csv_file(
        str(
            REPORTS_DIR
            / "final_test_predictions.csv"
        ),
        ["Date"]
    )

    validation_results = load_optional_csv(
        REPORTS_DIR
        / "stationary_model_validation_comparison.csv"
    )

    ablation_results = load_optional_csv(
        REPORTS_DIR
        / "sentiment_ablation_comparison.csv"
    )

    shap_importance = load_optional_csv(
        REPORTS_DIR
        / "final_model_shap_importance.csv"
    )

    movement_bundle, sentiment_pipeline = (
        load_project_models()
    )

except FileNotFoundError as error:
    st.error(
        "A required project file is missing:\n\n"
        f"{error}"
    )
    st.stop()


movement_model = movement_bundle["model"]
movement_feature_columns = movement_bundle[
    "feature_columns"
]


# ---------------------------------------------------------
# Feature engineering for dashboard predictions
# ---------------------------------------------------------

def prepare_prediction_features(dataframe):
    features = dataframe.copy()

    features["Date"] = pd.to_datetime(
        features["Date"]
    )

    features.sort_values(
        "Date",
        inplace=True
    )

    features.reset_index(
        drop=True,
        inplace=True
    )

    features["BB_Middle"] = (
        features["Close"]
        .rolling(window=20)
        .mean()
    )

    bb_standard_deviation = (
        features["Close"]
        .rolling(window=20)
        .std()
    )

    features["BB_Upper"] = (
        features["BB_Middle"]
        + (2 * bb_standard_deviation)
    )

    features["BB_Lower"] = (
        features["BB_Middle"]
        - (2 * bb_standard_deviation)
    )

    features["BB_Width"] = (
        (
            features["BB_Upper"]
            - features["BB_Lower"]
        )
        / features["BB_Middle"]
    )

    features["Close_vs_MA_5"] = (
        features["Close"]
        / features["MA_5"]
        - 1
    )

    features["Close_vs_MA_10"] = (
        features["Close"]
        / features["MA_10"]
        - 1
    )

    features["Close_vs_MA_20"] = (
        features["Close"]
        / features["MA_20"]
        - 1
    )

    features["MA_5_vs_MA_20"] = (
        features["MA_5"]
        / features["MA_20"]
        - 1
    )

    features["EMA_12_vs_EMA_26"] = (
        features["EMA_12"]
        / features["EMA_26"]
        - 1
    )

    features["MACD_Percentage"] = (
        features["MACD"]
        / features["Close"]
    )

    features["RSI_Normalized"] = (
        features["RSI_14"] - 50
    ) / 50

    features["BB_Position"] = (
        features["Close"]
        - features["BB_Lower"]
    ) / (
        features["BB_Upper"]
        - features["BB_Lower"]
    )

    features["Day_of_Week"] = (
        features["Date"].dt.dayofweek
    )

    features["Month"] = (
        features["Date"].dt.month
    )

    features["Day_Sin"] = np.sin(
        2
        * np.pi
        * features["Day_of_Week"]
        / 5
    )

    features["Day_Cos"] = np.cos(
        2
        * np.pi
        * features["Day_of_Week"]
        / 5
    )

    features["Month_Sin"] = np.sin(
        2
        * np.pi
        * features["Month"]
        / 12
    )

    features["Month_Cos"] = np.cos(
        2
        * np.pi
        * features["Month"]
        / 12
    )

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return features


prediction_data = prepare_prediction_features(
    market_data
)

valid_prediction_data = prediction_data.dropna(
    subset=movement_feature_columns
).copy()


# ---------------------------------------------------------
# Text-cleaning helper
# ---------------------------------------------------------

def clean_headline(text):
    cleaned_text = str(text).lower().strip()

    cleaned_text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        cleaned_text
    )

    cleaned_text = re.sub(
        r"[^a-z0-9\s%$.-]",
        " ",
        cleaned_text
    )

    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    ).strip()

    return cleaned_text


# ---------------------------------------------------------
# Reusable metrics
# ---------------------------------------------------------

test_accuracy = accuracy_score(
    test_results["Actual_Target"],
    test_results["Predicted_Target"]
)

test_precision = precision_score(
    test_results["Actual_Target"],
    test_results["Predicted_Target"],
    zero_division=0
)

test_recall = recall_score(
    test_results["Actual_Target"],
    test_results["Predicted_Target"],
    zero_division=0
)

test_f1 = f1_score(
    test_results["Actual_Target"],
    test_results["Predicted_Target"],
    zero_division=0
)

test_auc = roc_auc_score(
    test_results["Actual_Target"],
    test_results["Up_Probability"]
)


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------

st.sidebar.title("📈 AAPL Prediction")

st.sidebar.caption(
    "Financial market movement prediction using "
    "market indicators, NLP and explainable AI."
)

selected_page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Market Trends",
        "Sentiment Analysis",
        "Prediction Signal",
        "Model Performance",
        "Explainability",
        "Project Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Target: Predict whether Apple’s closing "
    "price will increase on the next trading day."
)

st.sidebar.warning(
    "Educational analytics only. "
    "This is not financial advice."
)


# ---------------------------------------------------------
# Overview page
# ---------------------------------------------------------

if selected_page == "Overview":
    st.title(
        "Apple Financial Market Movement Prediction"
    )

    st.markdown(
        """
        <div class="project-subtitle">
        An AI-powered analytics dashboard combining
        technical market indicators, financial-news
        sentiment and explainable machine learning.
        </div>
        """,
        unsafe_allow_html=True
    )

    latest_market_row = market_data.iloc[-1]
    previous_market_row = market_data.iloc[-2]

    latest_close = latest_market_row["Close"]

    latest_change = (
        (
            latest_close
            / previous_market_row["Close"]
        )
        - 1
    ) * 100

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Latest Closing Price",
        f"${latest_close:,.2f}",
        f"{latest_change:+.2f}%"
    )

    metric_2.metric(
        "Final Test Accuracy",
        f"{test_accuracy:.2%}"
    )

    metric_3.metric(
        "Final ROC-AUC",
        f"{test_auc:.2%}"
    )

    metric_4.metric(
        "News Headlines",
        f"{int(daily_sentiment['News_Count'].sum()):,}"
    )

    st.subheader("Apple Closing-Price History")

    recent_market = market_data.tail(750)

    overview_price_figure = px.line(
        recent_market,
        x="Date",
        y="Close",
        title=(
            "Recent Apple Closing-Price Movement"
        ),
        labels={
            "Close": "Closing Price (USD)",
            "Date": "Date"
        }
    )

    overview_price_figure.update_traces(
        line_color="#2563eb",
        line_width=2
    )

    overview_price_figure.update_layout(
        height=430,
        hovermode="x unified"
    )

    st.plotly_chart(
        overview_price_figure,
        use_container_width=True
    )

    st.subheader("Project Summary")

    summary_1, summary_2, summary_3 = (
        st.columns(3)
    )

    with summary_1:
        st.markdown("#### Market Analytics")
        st.write(
            "Daily returns, moving averages, RSI, "
            "MACD, Bollinger Bands and volatility."
        )

    with summary_2:
        st.markdown("#### Sentiment NLP")
        st.write(
            "TF-IDF and Logistic Regression classify "
            "headlines as positive, neutral or negative."
        )

    with summary_3:
        st.markdown("#### Explainable AI")
        st.write(
            "SHAP identifies the indicators influencing "
            "upward and downward predictions."
        )


# ---------------------------------------------------------
# Market trends page
# ---------------------------------------------------------

elif selected_page == "Market Trends":
    st.title("Apple Market Trends")

    minimum_date = market_data["Date"].min().date()
    maximum_date = market_data["Date"].max().date()

    default_start = (
        market_data["Date"].max()
        - pd.DateOffset(years=2)
    ).date()

    selected_dates = st.date_input(
        "Select the analysis period",
        value=(default_start, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date
    )

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):
        start_date, end_date = selected_dates
    else:
        start_date = default_start
        end_date = maximum_date

    filtered_market = market_data[
        (
            market_data["Date"].dt.date
            >= start_date
        )
        & (
            market_data["Date"].dt.date
            <= end_date
        )
    ].copy()

    st.subheader("Candlestick Chart")

    candlestick_figure = go.Figure(
        data=[
            go.Candlestick(
                x=filtered_market["Date"],
                open=filtered_market["Open"],
                high=filtered_market["High"],
                low=filtered_market["Low"],
                close=filtered_market["Close"],
                name="AAPL"
            )
        ]
    )

    candlestick_figure.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )

    st.plotly_chart(
        candlestick_figure,
        use_container_width=True
    )

    st.subheader("Moving Averages")

    moving_average_figure = go.Figure()

    moving_average_figure.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["Close"],
            name="Close",
            line=dict(
                color="#2563eb",
                width=2
            )
        )
    )

    moving_average_figure.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["MA_5"],
            name="5-Day MA",
            line=dict(
                color="#f97316",
                width=1.5
            )
        )
    )

    moving_average_figure.add_trace(
        go.Scatter(
            x=filtered_market["Date"],
            y=filtered_market["MA_20"],
            name="20-Day MA",
            line=dict(
                color="#16a34a",
                width=1.5
            )
        )
    )

    moving_average_figure.update_layout(
        height=440,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )

    st.plotly_chart(
        moving_average_figure,
        use_container_width=True
    )

    st.subheader("Trading Volume")

    volume_figure = px.bar(
        filtered_market,
        x="Date",
        y="Volume",
        labels={
            "Volume": "Trading Volume",
            "Date": "Date"
        }
    )

    volume_figure.update_traces(
        marker_color="#0f766e"
    )

    volume_figure.update_layout(
        height=350
    )

    st.plotly_chart(
        volume_figure,
        use_container_width=True
    )


# ---------------------------------------------------------
# Sentiment page
# ---------------------------------------------------------

elif selected_page == "Sentiment Analysis":
    st.title("Financial-News Sentiment Analysis")

    sentiment_history = modeling_history[
        [
            "Date",
            "Daily_Avg_Sentiment",
            "News_Count"
        ]
    ].copy()

    sentiment_history[
        "Rolling_Sentiment_30"
    ] = (
        sentiment_history[
            "Daily_Avg_Sentiment"
        ]
        .rolling(
            window=30,
            min_periods=1
        )
        .mean()
    )

    sentiment_history[
        "Rolling_News_Count_30"
    ] = (
        sentiment_history[
            "News_Count"
        ]
        .rolling(
            window=30,
            min_periods=1
        )
        .sum()
    )

    sentiment_figure = px.line(
        sentiment_history,
        x="Date",
        y="Rolling_Sentiment_30",
        title=(
            "Apple News Sentiment "
            "(30-Trading-Day Average)"
        ),
        labels={
            "Rolling_Sentiment_30":
                "Average Sentiment",
            "Date": "Date"
        }
    )

    sentiment_figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )

    sentiment_figure.update_traces(
        line_color="#9333ea",
        line_width=2
    )

    sentiment_figure.update_layout(
        height=420,
        hovermode="x unified"
    )

    st.plotly_chart(
        sentiment_figure,
        use_container_width=True
    )

    news_volume_figure = px.area(
        sentiment_history,
        x="Date",
        y="Rolling_News_Count_30",
        title=(
            "Apple News Volume "
            "(30-Trading-Day Total)"
        ),
        labels={
            "Rolling_News_Count_30":
                "Number of Headlines",
            "Date": "Date"
        }
    )

    news_volume_figure.update_traces(
        line_color="#0f766e",
        fillcolor="rgba(15, 118, 110, 0.25)"
    )

    news_volume_figure.update_layout(
        height=380
    )

    st.plotly_chart(
        news_volume_figure,
        use_container_width=True
    )

    st.subheader("Headline Sentiment Tester")

    user_headline = st.text_area(
        "Enter a financial-news headline",
        placeholder=(
            "Example: Apple reports stronger-than-"
            "expected quarterly revenue"
        )
    )

    if st.button(
        "Analyze Headline",
        type="primary"
    ):
        if not user_headline.strip():
            st.warning(
                "Please enter a headline."
            )
        else:
            cleaned_headline = clean_headline(
                user_headline
            )

            predicted_sentiment = (
                sentiment_pipeline.predict(
                    [cleaned_headline]
                )[0]
            )

            sentiment_probabilities = (
                sentiment_pipeline.predict_proba(
                    [cleaned_headline]
                )[0]
            )

            sentiment_classes = (
                sentiment_pipeline
                .named_steps["classifier"]
                .classes_
            )

            probability_mapping = dict(
                zip(
                    sentiment_classes,
                    sentiment_probabilities
                )
            )

            if predicted_sentiment == "positive":
                st.success(
                    "Predicted sentiment: POSITIVE"
                )
            elif predicted_sentiment == "negative":
                st.error(
                    "Predicted sentiment: NEGATIVE"
                )
            else:
                st.info(
                    "Predicted sentiment: NEUTRAL"
                )

            probability_1, probability_2, probability_3 = (
                st.columns(3)
            )

            probability_1.metric(
                "Negative Probability",
                (
                    f"{probability_mapping['negative']:.2%}"
                )
            )

            probability_2.metric(
                "Neutral Probability",
                (
                    f"{probability_mapping['neutral']:.2%}"
                )
            )

            probability_3.metric(
                "Positive Probability",
                (
                    f"{probability_mapping['positive']:.2%}"
                )
            )


# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------

elif selected_page == "Prediction Signal":
    st.title("Next-Trading-Day Prediction Signal")

    latest_prediction_row = (
        valid_prediction_data.iloc[[-1]]
    )

    latest_feature_values = (
        latest_prediction_row[
            movement_feature_columns
        ]
    )

    predicted_target = int(
        movement_model.predict(
            latest_feature_values
        )[0]
    )

    upward_probability = float(
        movement_model.predict_proba(
            latest_feature_values
        )[0, 1]
    )

    downward_probability = (
        1 - upward_probability
    )

    prediction_date = (
        latest_prediction_row[
            "Date"
        ].iloc[0]
    )

    latest_close = (
        latest_prediction_row[
            "Close"
        ].iloc[0]
    )

    st.caption(
        "Signal generated from the latest locally "
        "available market record."
    )

    signal_column, probability_column = (
        st.columns([1, 1])
    )

    with signal_column:
        st.metric(
            "Latest Market Date",
            prediction_date.strftime(
                "%d %B %Y"
            )
        )

        st.metric(
            "Latest Closing Price",
            f"${latest_close:,.2f}"
        )

        if predicted_target == 1:
            st.markdown(
                """
                <div class="signal-up">
                ▲ Predicted signal: UP
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="signal-down">
                ▼ Predicted signal: DOWN
                </div>
                """,
                unsafe_allow_html=True
            )

    with probability_column:
        probability_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=(
                    upward_probability * 100
                ),
                number={
                    "suffix": "%",
                    "valueformat": ".1f"
                },
                title={
                    "text":
                        "Probability of Upward Movement"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#2563eb"
                    },
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#fee2e2"
                        },
                        {
                            "range": [50, 100],
                            "color": "#dcfce7"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "black",
                            "width": 3
                        },
                        "value": 50
                    }
                }
            )
        )

        probability_gauge.update_layout(
            height=300,
            margin=dict(
                l=30,
                r=30,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            probability_gauge,
            use_container_width=True
        )

    probability_1, probability_2 = (
        st.columns(2)
    )

    probability_1.metric(
        "Down Probability",
        f"{downward_probability:.2%}"
    )

    probability_2.metric(
        "Up Probability",
        f"{upward_probability:.2%}"
    )

    st.subheader("Latest Technical Indicators")

    indicator_columns = [
        "Daily_Return",
        "RSI_Normalized",
        "MACD_Percentage",
        "BB_Position",
        "Volatility_10",
        "Close_vs_MA_20"
    ]

    indicator_table = (
        latest_prediction_row[
            indicator_columns
        ]
        .T
        .reset_index()
    )

    indicator_table.columns = [
        "Indicator",
        "Value"
    ]

    st.dataframe(
        indicator_table,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="information-box">
        The model provides a probabilistic analytical
        signal, not a guaranteed forecast. Market prices
        can change because of unexpected events that are
        not represented in historical data.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Model-performance page
# ---------------------------------------------------------

elif selected_page == "Model Performance":
    st.title("Model Performance")

    metric_1, metric_2, metric_3, metric_4, metric_5 = (
        st.columns(5)
    )

    metric_1.metric(
        "Accuracy",
        f"{test_accuracy:.2%}"
    )

    metric_2.metric(
        "Precision",
        f"{test_precision:.2%}"
    )

    metric_3.metric(
        "Recall",
        f"{test_recall:.2%}"
    )

    metric_4.metric(
        "F1-score",
        f"{test_f1:.2%}"
    )

    metric_5.metric(
        "ROC-AUC",
        f"{test_auc:.2%}"
    )

    evaluation_figure_path = (
        FIGURES_DIR
        / "final_model_evaluation.png"
    )

    if evaluation_figure_path.exists():
        st.image(
            str(evaluation_figure_path),
            caption=(
                "Final confusion matrix and ROC curve"
            ),
            use_container_width=True
        )

    st.subheader(
        "Validation Comparison of Four Models"
    )

    if not validation_results.empty:
        st.dataframe(
            validation_results.round(4),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info(
            "Validation comparison file is unavailable."
        )

    st.subheader(
        "Market-Only vs Sentiment-Enhanced Model"
    )

    if not ablation_results.empty:
        st.dataframe(
            ablation_results.round(4),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "The market-only model generalized better "
            "than the sentiment-enhanced model."
        )
    else:
        st.info(
            "Sentiment ablation results are unavailable."
        )

    st.subheader("Recent Test Predictions")

    recent_test_predictions = (
        test_results.tail(20).copy()
    )

    recent_test_predictions[
        "Actual_Movement"
    ] = recent_test_predictions[
        "Actual_Target"
    ].map({
        0: "Down",
        1: "Up"
    })

    recent_test_predictions[
        "Predicted_Movement"
    ] = recent_test_predictions[
        "Predicted_Target"
    ].map({
        0: "Down",
        1: "Up"
    })

    st.dataframe(
        recent_test_predictions[
            [
                "Date",
                "Actual_Movement",
                "Predicted_Movement",
                "Up_Probability"
            ]
        ].sort_values(
            "Date",
            ascending=False
        ),
        hide_index=True,
        use_container_width=True
    )


# ---------------------------------------------------------
# Explainability page
# ---------------------------------------------------------

elif selected_page == "Explainability":
    st.title("Explainable AI with SHAP")

    st.write(
        "SHAP explains how each feature influences "
        "the model’s predictions. Larger absolute SHAP "
        "values indicate stronger influence."
    )

    final_shap_path = (
        FIGURES_DIR
        / "final_model_shap_summary.png"
    )

    if final_shap_path.exists():
        st.subheader(
            "Final Model Feature Effects"
        )

        st.image(
            str(final_shap_path),
            caption=(
                "SHAP summary for the final "
                "market-only Logistic Regression model"
            ),
            use_container_width=True
        )

    if not shap_importance.empty:
        st.subheader(
            "Top Feature-Importance Ranking"
        )

        st.dataframe(
            shap_importance.head(15).round(6),
            hide_index=True,
            use_container_width=True
        )

    sentiment_shap_path = (
        FIGURES_DIR
        / "sentiment_shap_impact.png"
    )

    if sentiment_shap_path.exists():
        st.subheader(
            "Impact of Sentiment Features"
        )

        st.image(
            str(sentiment_shap_path),
            caption=(
                "SHAP importance of sentiment features "
                "in the combined candidate model"
            ),
            use_container_width=True
        )

        st.info(
            "Sentiment features represented 30.31% of "
            "combined-model SHAP importance, but they "
            "reduced out-of-sample predictive accuracy."
        )


# ---------------------------------------------------------
# Project-information page
# ---------------------------------------------------------

elif selected_page == "Project Information":
    st.title("Project Information")

    st.markdown(
        """
        ### Project objective

        Predict whether Apple’s closing price will move
        upward or downward on the following trading day.

        ### Data sources

        - Yahoo Finance historical Apple market data
        - Financial PhraseBank labelled sentiment data
        - Historical Apple financial-news headlines

        ### Market features

        - Daily returns and lagged returns
        - Moving averages and exponential averages
        - RSI and MACD
        - Bollinger Bands
        - Price range and trading-volume changes
        - Rolling volatility and calendar variables

        ### NLP pipeline

        1. Clean financial-news headlines
        2. Convert text into TF-IDF features
        3. Classify sentiment using Logistic Regression
        4. Generate positive, neutral and negative scores
        5. Aggregate sentiment by trading day

        ### Machine-learning models compared

        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM

        ### Final model

        The final model is a regularized Logistic
        Regression classifier using stationary,
        scale-independent market features. It was selected
        using chronological validation without using the
        test period for model selection.

        ### Key finding

        Technical indicators produced a modest predictive
        signal. Adding historical news sentiment reduced
        out-of-sample performance because news coverage was
        inconsistent and sentiment was often reactive rather
        than predictive.

        ### Limitations

        - Financial markets contain substantial randomness.
        - Historical relationships can change over time.
        - News coverage is sparse in earlier years.
        - Transaction costs and slippage are not included.
        - The application is educational and is not
          financial advice.
        """
    )