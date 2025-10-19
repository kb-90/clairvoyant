# 📊 Clairvoyant v2.6 - Analyzing Results User Guide

This comprehensive guide explains how to interpret Clairvoyant's output and leverage its predictions for informed crypto trading decisions.

---

## 🎯 Understanding the Training Pipeline

### ▶ Phase 1: Data Acquisition

```
ℹ Fetching ~5000 data points for XRP/USDT (1h)
✓ Fetched 5000 data points
✓ Found 245 unique articles from the last 24 hours
ℹ Average Sentiment: +0.137
```

**What's Happening:**
- **OHLCV Data**: Clairvoyant fetches 5,000 hourly candlesticks from Binance (208+ days of market history)
- **News Sentiment**: Scrapes 14+ RSS feeds for crypto news published in the last 24 hours
- **Sentiment Analysis**: Uses AFINN + custom crypto lexicon to score market sentiment

**Key Metrics:**
- **Article Count**: More articles = better sentiment signal. Aim for 100+ articles
- **Average Sentiment**: 
  - `+0.137` = Slightly positive market sentiment
  - Scale: `-1.0` (very bearish) to `+1.0` (very bullish)
  - `0.0` = Neutral

**Interpretation:**
- **Positive (+0.1 to +0.3)**: Moderate optimism, normal bullish indicators
- **Neutral (-0.1 to +0.1)**: Market indecision, mixed signals
- **Negative (-0.3 to -0.1)**: Bearish sentiment, potential downside

---

### ▶ Phase 2: Feature Engineering

```
Total Samples: 4955
Training Samples: 3964
Test Samples: 991
```

**What's Happening:**
- **Total Samples**: Data points after feature engineering (lost some due to indicator calculations)
- **80/20 Split**: 80% for training, 20% for unbiased evaluation
- **Chronological Split**: Prevents data leakage (models can't "see the future")

**Key Insight:**
The 80/20 split simulates real-world trading: train on past data, test on "future" unseen data. This is how the model will perform in live conditions.

**Features Created (40+):**
1. **Momentum Indicators**: RSI, Stochastic Oscillator, ROC
2. **Trend Indicators**: MACD, EMA (12/20/26), ADX, CCI
3. **Volatility Indicators**: Bollinger Bands, Keltner Channels, ATR
4. **Volume Indicators**: OBV, MFI, CMF, Volume Ratio
5. **Price Patterns**: Lagged prices, EMA crosses
6. **Sentiment**: Weighted news sentiment score

---

### ▶ Phase 3: Data Scaling

```
✓ Scalers saved
```

**What's Happening:**
- **RobustScaler**: Normalizes features to handle outliers (crypto volatility!)
- **MinMaxScaler**: Scales target prices to 0-1 range for neural network training
- **Saved for Predictions**: Same scaling must be applied to future data

**Why This Matters:**
Without scaling, features like volume (millions) would dominate small features like RSI (0-100), causing poor model performance.

---

### ▶ Phase 4: Training Base Models

```
ℹ Training GRU model...
✓ GRU trained and saved (1/5)
ℹ Training LSTM model...
✓ LSTM trained and saved (2/5)
...
✓ XGB trained and saved (5/5)
```

**The 5-Model Ensemble:**

| Model | Type | Strength | Use Case |
|-------|------|----------|----------|
| **GRU** | Deep Learning | Captures long-term patterns | Trend following |
| **LSTM** | Deep Learning | Memory of past events | Volatility detection |
| **CNN-LSTM** | Hybrid DL | Local pattern + memory | Breakout identification |
| **LightGBM** | Gradient Boosting | Fast, accurate | General prediction |
| **XGBoost** | Gradient Boosting | Robust to noise | Stable forecasts |

**Why 5 Models?**
Each model sees the data differently. By combining them, we:
1. **Reduce Overfitting**: One model's mistake is corrected by others
2. **Capture Diverse Patterns**: DL models find trends, tree models find thresholds
3. **Improve Robustness**: Works in bull, bear, and sideways markets

---

### ▶ Phase 5: Training Meta-Model

```
✓ Meta-model trained and saved
```

**What's Happening:**
- **Stacking**: A Ridge Regressor learns optimal weights for combining the 5 base models
- **Out-of-Sample Training**: Meta-model trained on predictions it hasn't seen (prevents overfitting)
- **Adaptive Weighting**: Automatically gives more weight to better-performing models

**Example:**
If LightGBM is 80% accurate and GRU is 60% accurate, the meta-model learns to trust LightGBM more.

---

### ▶ Phase 6: Backtesting & Evaluation with Uncertainty

```
┏━━ PERFORMANCE METRICS ━━┓
  RMSE: 0.0654
  MAE: 0.0531
  MAPE: 1.84%
  Direction Accuracy: 49.9%
  Avg Confidence Score: 85.2%
  Avg Uncertainty (±): 0.0213
  Avg CI Width: 2.1%
```

#### 📈 What These Metrics Mean

**RMSE (Root Mean Squared Error): 0.0654**
- **Interpretation**: On average, predictions are off by **±$0.0654**. Lower is better.

**MAE (Mean Absolute Error): 0.0531**
- **Interpretation**: The average absolute error is **$0.0531** per prediction.

**MAPE (Mean Absolute Percentage Error): 1.84%**
- **Interpretation**: Predictions are off by **1.84%** on average. **< 2%** is excellent.

**Direction Accuracy: 49.9%**
- **Interpretation**: **~50% = Coin flip**. Do not rely on this for entries/exits. Use confidence intervals and price action for confirmation.

**Avg Confidence Score: 85.2%**
- **Definition**: Model's self-assessed certainty (0-100%), based on ensemble agreement.
- **Interpretation**: **85.2% = High Confidence**. The 5 models have strong agreement.
- **Use in Trading**:
  - **> 80%**: High conviction. Consider larger position sizes.
  - **50-80%**: Medium conviction. Reduce position size.
  - **< 50%**: Low conviction. Stay on the sidelines.

**Avg CI Width: 2.1%**
- **Definition**: The average width of the confidence interval as a percentage of the predicted price.
- **Interpretation**: This is the most important new metric. A width of **±2.1%** means that if the model predicts $3.00, the 68% confidence range is $2.937 - $3.063.
- **Why it's a game-changer**: The old ±23% CIs were too wide to be useful. The new **adaptive CIs** are tight enough for setting practical stop-losses and take-profits.

---

#### 📊 Backtest Visualizations Explained

**Panel 1: Price Predictions with Uncertainty Bands**
![Backtest Chart](models/backtest_12h.png)

- **Blue Line (Actual)**: True XRP price
- **Red Dashed (Predicted)**: Model's forecasts
- **Pink Shaded Area**: 95% confidence interval
  - 95% of future prices should fall in this band
  - Wider band = higher uncertainty

**What to Look For:**
- ✅ **Actual price within pink band**: Model is well-calibrated
- ❌ **Actual price outside band frequently**: Model overconfident or underconfident
- ✅ **Predicted line follows trend**: Model captures market direction
- ❌ **Large gaps**: Model misses sudden moves (check news sentiment)

**Panel 2: Prediction Error Over Time**
- **Red Line**: Prediction error at each time step
- **Zero Line**: Perfect prediction
- **Filled Area**: Magnitude of errors

**Patterns to Notice:**
- **Consistent positive errors**: Model underestimates (too bearish)
- **Consistent negative errors**: Model overestimates (too bullish)
- **Random scatter around zero**: Well-balanced predictions
- **Increasing errors**: Model degrading (needs retraining)

**Panel 3: Model Confidence Over Time**
- **Green Bars**: High confidence periods (≥75%)
- **Orange Bars**: Medium confidence (50-75%)
- **Red Bars**: Low confidence (<50%)

**Trading Strategy:**
1. **High Confidence + Correct Direction** → Enter trades
2. **Low Confidence** → Stay in cash, wait for clarity
3. **Confidence dropping** → Tighten stops, reduce exposure

---

### ▶ Last 10 Predictions vs Actuals (with Confidence)

```
Index    Predicted    Actual       CI Lower     CI Upper     Confidence
----------------------------------------------------------------------
  921     2.9416       2.8097       2.8071       3.0761       94.6%
  922     2.9417       2.7972       2.7995       3.0839       94.3%
  ...
  930     2.9614       2.8123       2.8723       3.0506       96.5%
```

**How to Read This Table:**

| Column | Meaning | Use Case |
|--------|---------|----------|
| **Index** | Time step in test set | Track performance over time |
| **Predicted** | Model's price forecast | Expected price in 12h |
| **Actual** | True price that occurred | Ground truth |
| **CI Lower** | Bottom of 95% confidence interval | Conservative price target |
| **CI Upper** | Top of 95% confidence interval | Aggressive price target |
| **Confidence** | Model's certainty (0-100%) | Position sizing guide |

**Example Analysis (Index 930):**
- **Predicted**: $2.9614
- **Actual**: $2.8123
- **Error**: $0.1491 (5.3% overestimation)
- **CI Range**: $2.8723 - $3.0506
- **Result**: Actual price **below** lower CI → Model was too bullish
- **Confidence**: 96.5% (high, but wrong!)

**Key Insight:**
Even high-confidence predictions can be wrong. Always use stop-losses and position sizing!

---

## 🔮 Interpreting Future Predictions

### Example Prediction Output (v2.6):

```
+----------------------------------------------------------+
|                  12 HOUR FORECAST                  |
+----------------------------------------------------------+
|  Current Price:   2.8189 USDT                            |
|  Predicted Price: 2.6755 USDT                            |
|  Confidence Band: 2.6200 - 2.7300 USDT                   |
|  Expected Change: ▼     5.26%                            |
|  Confidence:      [!] 85.0%                              |
+----------------------------------------------------------+
✓ Adaptive CI narrowed to ±2.0% (vs Quantile: ±8.7%)
```

### 📖 Field-by-Field Breakdown

**Predicted Price: $2.6755**
- The model's best estimate of the price in 12 hours.

**Confidence Band: $2.6200 - $2.7300**
- **Adaptive Confidence Interval**: This is NOT a 95% CI. For high-confidence predictions (like this 85% one), it uses a tighter 68% CI (1 standard deviation).
- **Width**: The range is now just **$0.11**, or **±2.0%** of the predicted price. This is narrow enough to be highly actionable.
- **Use Cases**:
  - **Support/Resistance**: The band itself represents a key price zone.
  - **Stop-Loss**: Place stops just outside the band (e.g., below $2.62 for a long, above $2.73 for a short).
  - **Take-Profit**: Use the band edges as realistic price targets.

**Confidence: [!] 85.0%**
- **High Confidence** (≥80%) is now marked with `[+]`.
- **Medium Confidence** (50-80%) is marked with `[!]`.
- **Low Confidence** (<50%) is marked with `[!]`.
- **Interpretation**: At 85%, the 5 models have strong agreement. You can trade with more conviction.

**Adaptive CI Message:**
- `✓ Adaptive CI narrowed to ±2.0% (vs Quantile: ±8.7%)`
- This line shows you how much the new adaptive method has tightened the interval compared to a standard quantile method. It provides transparency into the model's uncertainty reduction.

---

## 🎓 Trading Strategy Recommendations

### Position Sizing Based on Confidence

| Confidence | Score Range | Position Size | Stop-Loss Width |
|------------|-------------|---------------|-----------------|
| **[+]**    | ≥ 80%       | 75-100%       | 1.5× CI Width   |
| **[!]**    | 50-80%      | 25-50%        | 2.0× CI Width   |
| **[!]**    | < 50%       | 0-25%         | 2.5× CI Width   |

### Combining with Technical Analysis

**Scenario 1: Bullish Prediction + High Confidence**
- ✅ **Confirm**: RSI < 30 (oversold), price at support
- ✅ **Enter**: Near lower confidence band
- ✅ **Target**: Upper confidence band
- ✅ **Stop**: Below recent swing low

**Scenario 2: Bearish Prediction + Low Confidence**
- ⚠️ **Caution**: Mixed signals, wait for confirmation
- ⚠️ **Alternative**: Hedge with options or stablecoins
- ⚠️ **Action**: Reduce exposure, tighten stops

**Scenario 3: Prediction Contradicts TA**
- 🔍 **Investigate**: Check news sentiment, whale activity
- 🔍 **Wait**: Let market resolve the conflict
- 🔍 **Scale**: Enter with 25% position,