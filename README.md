# CLAIRVOYANT v3.1 - XRP PRICE PREDICTOR

<p>
  <img src="/assets/clairvoyant-banner.jpg?text=Clairvoyant+Project+Banner" alt="Clairvoyant Banner">
</p>

<p align="left">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

**Clairvoyant** is a sophisticated cryptocurrency price forecaster that synthesizes market data, news sentiment, and dynamic on-chain metrics using a powerful stacking ensemble model.

Version 3.1 introduces a robust, dynamic on-chain analysis engine that moves beyond static metrics to track the real-time activity of the wealthiest XRP accounts, providing a true insight into market-moving whale behavior.

## Table of Contents
- [Why Clairvoyant?](#why-clairvoyant)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

## Why Clairvoyant?

Clairvoyant is designed to be both powerful and exceptionally easy to use. It bridges the gap between complex, institutional-grade forecasting models and the need for a simple, 'plug-and-play' user experience.

-   **Zero API Key Hassle:** Get started in minutes. Clairvoyant uses public, free-to-use APIs for all its data fetching (market, news, and on-chain). You don't need to sign up for any services, create API keys, or manage credentials.

-   **CPU & GPU Compatible:** Clairvoyant was built with accessibility in mind and runs efficiently on standard CPU-only machines. While a GPU is not required, users with a compatible NVIDIA GPU and the appropriate TensorFlow build will benefit from significantly faster model training times, allowing for quicker iterations and a more fluid data analysis experience with tools like TensorBoard.

-   **All-in-One Powerhouse:** While the setup is simple, the engine is not. Clairvoyant is a highly efficient script that packs a multi-model deep learning ensemble, real-time news sentiment analysis, and dynamic on-chain whale tracking into a single, cohesive pipeline.

-   **Simple & Centralized Configuration:** All user-adjustable settings are located in one place: the `.env` file. Tweak the target crypto, prediction horizons, or training parameters without ever needing to touch the core application logic.

-   **Transparent & Understandable:** The core logic is contained within a single, well-documented Python file (`clairvoyant_v3-1.py`). This makes it easy for developers to understand, audit, and extend the model's capabilities.

## Technology Stack

- **Languages:** Python (3.10+)
- **Machine Learning:** TensorFlow (Keras), Scikit-learn, LightGBM, XGBoost
- **Data & Analysis:** Pandas, NumPy, TA (Technical Analysis)
- **Data Fetching:** CCXT (Exchanges), aiohttp (Async Web), feedparser (RSS)
- **NLP & Sentiment:** NLTK, Afinn

## Project Architecture

The Clairvoyant architecture is built on three core data pillars that feed into a sophisticated stacking ensemble model.

<p align="center">
  <em>Data Pillars → Feature Engineering → Stacking Ensemble → Prediction</em>
</p>

1.  **Market Data Pillar**: Fetches thousands of historical OHLCV (Open, High, Low, Close, Volume) data points from Binance via the `ccxt` library to build a baseline of price action.

2.  **Sentiment Data Pillar**: Scrapes and analyzes news articles from over 14 financial RSS feeds. Using a custom crypto-specific lexicon, it generates a sentiment score that gauges the mood of the market.

3.  **On-Chain Data Pillar (Dynamic)**: This is the key innovation in v3.1. The system dynamically identifies the top 50 "whale" accounts by querying the `xrpscan.com` API. It then analyzes the last 12 hours of transactions for these accounts to detect significant market pressure from major players.

These three data sources are processed into a rich feature set and fed into a two-layer stacking ensemble:

-   **Base Models**: A diverse set of five models (Bi-GRU, Bi-LSTM, CNN-LSTM, LightGBM, XGBoost) that capture different types of patterns in the data.
-   **Meta-Model**: A `Ridge` regressor that intelligently combines the predictions from the base models into a single, more accurate, and robust final forecast.

## Key Features

-   **Dynamic On-Chain Analysis**: Moves beyond static metrics by identifying and tracking the activity of the top 50 XRP accounts in real-time.
-   **Stacking Ensemble Model**: Combines deep learning (for temporal patterns) and gradient-boosted trees (for non-linear relationships) for state-of-the-art prediction accuracy.
-   **Real-Time News Sentiment**: Integrates sentiment from over a dozen financial news sources using a crypto-specific lexicon for enhanced context.
-   **Adaptive Confidence Intervals**: Instead of a single price target, Clairvoyant provides a statistically-driven price *range* to express the uncertainty of its forecast. This range is dynamic, adapting to market conditions:
    - In stable periods, the confidence interval may be as tight as **±2%**, providing an actionable range for setting precise trading targets.
    - In volatile or uncertain conditions, the interval will widen (e.g., towards **±5%** or more), signaling a higher margin of error and advising a more cautious approach. This adaptive range is more reliable than a single point prediction and serves as a built-in risk indicator.
-   **Multi-Horizon Forecasting**: Trains specialized models to predict prices for various time horizons.
-   **Automated Pipeline**: Fully automated process from data fetching and feature engineering to training and prediction.

## Getting Started

Follow these steps to get Clairvoyant running on your local machine.

> [!NOTE]
> Python 3.10 or higher is recommended.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/clairvoyant.git
    cd clairvoyant
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the script**
    All user-adjustable parameters are located in the `.env` file. Open this file and modify the values to tune the model's behavior before running.

4.  **Run the Script**
    Execute the main script from your terminal:
    ```bash
    python clairvoyant_v3-1.py
    ```
    The script will handle data fetching, training, and prediction automatically. Results, models, and logs will be saved in their respective directories (`/predictions`, `/models`, `/logs`).

## Configuration

All configuration is handled in the `.env` file. Here is a description of the available parameters:

| Parameter                  | Description                                                                 | Default      | Recommendation / Notes                                       |
| -------------------------- | --------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------ |
| `TICKER`                   | The cryptocurrency pair to trade (as recognized by Binance).                | `XRP/USDT`   | Any valid `ccxt` pair (e.g., `BTC/USDT`, `ETH/USDT`).          |
| `TIMEFRAME`                | The candle timeframe for the data.                                          | `1h`         | Standard `ccxt` timeframes (e.g., `30m`, `4h`, `1d`).          |
| `DATA_LIMIT`               | The number of historical data points (candles) to fetch for training.       | `5000`       | More data can improve accuracy but increases training time.  |
| `PREDICTION_HORIZONS`      | A comma-separated list of future hours to predict.                          | `48`         | e.g., `6,24,48`. The script will train a separate model for each horizon. |
| `SEQUENCE_LENGTH`          | The number of past time steps to use as input for the deep learning models. | `60`         | Should not be changed unless you have a deep understanding of LSTMs. |
| `OPTIMIZE_HYPERPARAMETERS` | Set to `True` to run a lengthy Optuna study to find the best model params.  | `False`      | Recommended to keep `False` for daily runs. Use only for periodic deep tuning. | 
| `OPTUNA_TRIALS`            | The number of trials to run during the optimization study.                  | `50`         | Only active if the above is `True`. More trials can find better models but takes longer. |

## Project Structure

```text
clairvoyant/
├── .env                  # User-configurable parameters for the script
├── clairvoyant_v3-1.py   # The core script containing all logic
├── requirements.txt      # Project dependencies
├── README.md             # This file
├── documentation/        # Contains user guides and development notes
├── dtbs/                 # Database files for Optuna hyperparameter studies
├── lexicon/              # Custom sentiment dictionary for crypto terms
│   └── crypto_lexicon.py
├── logs/                 # Stores timestamped TensorBoard logs for each run
│   └── run_YYYYMMDD_HHMMSS/
├── models/               # Stores trained model files (.pkl)
│   ├── base_cnn_lstm_[x]h.pkl
│   └── meta_model_[x]h.pkl
├── predictions/          # Logs predictions and backtest plots
│   ├── predictions.csv
│   └── backtest_[x]h.png
└── sentiment/            # Caches sentiment data and trend plots
    ├── news_sentiment_XRP.csv
    └── sentiment_trend_XRP.png
```

## Disclaimer

This project and its predictions are for educational and informational purposes only. Cryptocurrency markets are extremely volatile. Always conduct your own research and do not consider this as financial advice. Past performance is not indicative of future results.