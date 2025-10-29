# CLAIRVOYANT v3.2 - XRP PRICE FORECASTER

<img src="/assets/clairvoyant-banner.jpg" alt="Clairvoyant Banner">

<p align="center">
  <a href="https://www.python.org/" alt="Badge: Python 3.10+"><img src="https://img.shields.io/badge/-3.10%2B-grey?style=flat-square&logo=python&logoColor=white&labelColor=%233776AB"></a>
  <a href="https://opensource.org/licenses/MIT" alt="License: MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>
  <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional" alt="Badge: Bi-GRU"><img src="https://img.shields.io/badge/ML-BiGRU-grey?style=flat-square&labelColor=purple"></a>
  <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM" alt="Badge: LSTM"><img src="https://img.shields.io/badge/ML-LSTM-grey?style=flat-square&labelColor=magenta"></a>
  <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM1D" alt="Badge: CNN-LSTM"><img src="https://img.shields.io/badge/ML-CNN--LSTM-grey?style=flat-square&labelColor=red"></a>
  <a href="https://lightgbm.readthedocs.io/en/latest/" alt="Badge: LightGBM"><img src="https://img.shields.io/badge/ML-LightGBM-grey?style=flat-square&labelColor=yellow"></a>
  <a href="https://xgboost.readthedocs.io/en/stable/" alt="Badge: XGBoost"><img src="https://img.shields.io/badge/ML-XGBoost-grey?style=flat-square&labelColor=limegreen" /></a>
  <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html" alt="Badge: Ridge Regression"><img src="https://img.shields.io/badge/ML-Ridge--Regression-grey?style=flat-square&labelColor=turquiose"></a>
  <a href="https://en.wikipedia.org/wiki/RSS" alt="Badge: RSS"><img src="https://img.shields.io/badge/RSS-FFA500?style=flat-square&logo=rss&logoColor=white"></a>
  <a href="httpshttps://github.com/ccxt/ccxt" alt="Badge: CCXT"><img src="https://img.shields.io/badge/-CCXT-black?style=flat-square&logo=x&logoColor=white&labelColor=black"></a>
  <a href="https://www.tensorflow.org/tensorboard" alt="Badge: TensorBoard"><img src="https://img.shields.io/badge/-TensorBoard-orange?style=flat-square&logo=tensorflow&logoColor=white&labelColor=orange"></a>
  <a href="https://xrpscan.com/" alt="Badge: XRPScan"><img src="https://img.shields.io/badge/XRPScan-black?style=flat-square&logo=xrp&logoColor=white"></a>
  <a href="https://www.paypal.com/paypalme/kb90fund" alt="Badge: Fund my dev"><img src="https://img.shields.io/badge/-support_me-blue?style=flat-square&logo=paypal&logoColor=white"></a>
</p>

**Clairvoyant** is a sophisticated cryptocurrency price forecaster that synthesizes market data, news sentiment, and dynamic on-chain metrics using a powerful stacking ensemble model.

Version 3.2 introduces significant performance enhancements, including a faster 80/20 train/test split and improved charting functionalities. It builds upon the robust, dynamic on-chain analysis engine from v3.1 that moves beyond static metrics to track the real-time activity of the wealthiest XRP accounts, providing a true insight into market-moving whale behavior.

## Table of Contents

- [Why Clairvoyant?](#why-clairvoyant)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Acknowledgements & Licenses](#acknowledgements--licenses)
- [Disclaimer](#disclaimer)

## Why Clairvoyant?

Clairvoyant is designed to be both powerful and exceptionally easy to use. It bridges the gap between complex, institutional-grade forecasting models and the need for a simple, 'plug-and-play' user experience.

- **Zero API Key Hassle:** Get started in minutes. Clairvoyant uses public, free-to-use APIs for all its data fetching (market, news, and on-chain). You don't need to sign up for any services, create API keys, or manage credentials.
- **CPU & GPU Compatible:** Clairvoyant was built with accessibility in mind and runs efficiently on standard CPU-only machines. While a GPU is not required, users with a compatible NVIDIA GPU and the appropriate TensorFlow build will benefit from significantly faster model training times, allowing for quicker iterations and a more fluid data analysis experience with tools like TensorBoard.
- **All-in-One Powerhouse:** While the setup is simple, the engine is not. Clairvoyant is a highly efficient script that packs a multi-model deep learning ensemble, real-time news sentiment analysis, and dynamic on-chain whale tracking into a single, cohesive pipeline.
- **Simple & Centralized Configuration:** All user-adjustable settings are located in one place: the `.env` file. Tweak the target crypto, prediction horizons, or training parameters without ever needing to touch the core application logic.
- **Transparent & Understandable:** The core logic is contained within a single, well-documented Python file (`clairvoyant_v3-1.py`). This makes it easy for developers to understand, audit, and extend the model's capabilities.

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

*1* **Market Data Pillar**
  
- Fetches thousands of historical OHLCV (Open, High, Low, Close, Volume) data points from Binance via the `ccxt` library to build a baseline of price action.

*2* **Sentiment Data Pillar**
  
- Scrapes and analyzes news articles from over 14 financial RSS feeds. Using a custom crypto-specific lexicon, it generates a sentiment score that gauges the mood of the market.

*3* **On-Chain Data Pillar (Dynamic)**

- This is the key innovation introduced in v3.1. The system dynamically identifies the top 50 "whale" accounts by querying the `xrpscan.com` API.
- It then analyzes the last 12 hours of transactions for these accounts to detect significant market pressure from major players.

### These three data sources are processed into a rich feature set and fed into a two-layer stacking ensemble

- ***Base Models***: A diverse set of five models (Bi-GRU, Bi-LSTM, CNN-LSTM, LightGBM, XGBoost) that capture different types of patterns in the data.
- ***Meta-Model***: A `Ridge` regressor that intelligently combines the predictions from the base models into a single, more accurate, and robust final forecast.

## Key Features

- **Dynamic On-Chain Analysis**: Moves beyond static metrics by identifying and tracking the activity of the top 50 XRP accounts in real-time.
- **Stacking Ensemble Model**: Combines deep learning (for temporal patterns) and gradient-boosted trees (for non-linear relationships) for state-of-the-art prediction accuracy.
- **Real-Time News Sentiment**: Integrates sentiment from over a dozen financial news sources using a crypto-specific lexicon for enhanced context.
- **Adaptive Confidence Intervals**: Instead of a single price target, Clairvoyant provides a statistically-driven price *range* to express the uncertainty of its forecast.
- **This range is dynamic, adapting to market conditions:**
  - In stable periods, the confidence interval may be as tight as **±2%**, providing an actionable range for setting precise trading targets.
  - In volatile or uncertain conditions, the interval will widen (e.g., towards **±5%** or more), signaling a higher margin of error and advising a more cautious approach.
  - This adaptive range is more reliable than a single point prediction and serves as a built-in risk indicator.
- **Multi-Horizon Forecasting**: Trains specialized models to predict prices for various time horizons.
- **Automated Pipeline**: Fully automated process from data fetching and feature engineering to training and prediction.

<p align="center">
  <img width="49%" src="/assets/future_forecast_readme_2.png" alt="Clairvoyant XRP Price Forecaster - Predicted Plot">
  <img width="49%" src="/assets/sentiment_trend_XRP_readme.png" alt="Clairvoyant XRP Price Forecaster - XRP Sentiment Trend Analysis Chart">
</p>
 
## Getting Started

Follow these steps to get Clairvoyant running on your local machine.

*1.***Clone the Repository**

  ```bash
  git clone https://github.com/your-username/clairvoyant.git
  cd clairvoyant
  ```

*2.***Install Dependencies**

  ```bash
  pip install -r requirements.txt
  ```

*3.***Configure the script**
  All user-adjustable parameters are located in the `.env` file. Open this file and modify the values to tune the model's behavior before running.

*4.***Run the Script**
  Execute the main script from your terminal:

  ```bash
  python clairvoyant_v3-2.py
  ```

  The script will handle data fetching, training, and prediction automatically. 
  Results, models, and logs will be saved in their respective directories (`/predictions`, `/models`, `/logs`).

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
├── clairvoyant_v3-2.py   # The core script containing all logic
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
│   ├── future_forecast_1.png
    ├── future_forecast_2.png
    └── future_forecast_3.png
└── sentiment/            # Caches sentiment data and trend plots
    ├── news_sentiment_XRP.csv
    └── sentiment_trend_XRP.png
```

## Acknowledgements & Licenses

This project utilizes several open-source libraries. We are grateful to the developers and contributors of these projects.

- **AFINN**: Licensed under the Apache 2.0 License.
- **CCXT**: Licensed under the MIT License.
- **NLTK (Natural Language Toolkit)**: Licensed under the Apache 2.0 License.
- **aiohttp**: Licensed under the Apache 2.0 License.
- **pandas**: Licensed under the BSD 3-Clause License.
- **joblib**: Licensed under the BSD 3-Clause License.
- **XGBoost**: Licensed under the Apache 2.0 License.
- **ta**: Licensed under the MIT License.

> [NOTE] *This list is not exhaustive.
> For a complete list of dependencies and their licenses, you can use tools like `pip-licenses`.*

> [WARNING] ## Disclaimer
> This project and its predictions are for educational and informational purposes only. Cryptocurrency markets are extremely volatile.
> Always conduct your own research and do not consider this as financial advice. Past performance is not indicative of future results.
