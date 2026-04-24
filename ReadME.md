# 📈 LSTM Stock Price Predictor

> A deep-learning pipeline that forecasts equity prices using stacked LSTM networks trained on historical OHLCV data from Yahoo Finance.

---

## 🧠 Project Overview

This project builds a production-style time-series forecasting system for stock prices.  
A **3-layer stacked LSTM** model is trained on several years of daily closing prices, then evaluated against a hold-out test period and used to generate a **next-trading-day price forecast**.

The project demonstrates practical skills in:

- Deep learning with **TensorFlow / Keras** (LSTM, Dropout, Dense)
- Time-series feature engineering (sliding-window sequences, MinMax scaling)
- End-to-end ML pipeline: data ingestion → preprocessing → training → evaluation → inference
- Model evaluation with **MAE, RMSE, MAPE, R²**
- Data visualisation with **Matplotlib** (dark-theme multi-panel chart)
- CLI design with `argparse` for reusable tooling

---

## 🗂️ Repository Structure

```
stock_predictor/
├── stock_predictor.py   # Full pipeline (data → model → forecast)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Architecture

```
Input sequence  (60 days × 1 feature)
        │
   LSTM(50) → Dropout(0.2)
        │
   LSTM(50) → Dropout(0.2)
        │
   LSTM(50) → Dropout(0.2)
        │
    Dense(1)
        │
  Predicted price (t+1)
```

| Hyperparameter    | Value |
|-------------------|-------|
| Look-back window  | 60 days |
| LSTM units / layer | 50 |
| Dropout rate      | 0.2 |
| Optimizer         | Adam |
| Loss              | MSE |
| Batch size        | 32 |
| Default epochs    | 25 |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with defaults  *(Google / GOOG, 2014–2023 training)*
```bash
python stock_predictor.py
```

### 3. Custom ticker & date range
```bash
python stock_predictor.py --ticker AAPL --train-start 2015-01-01 --train-end 2023-06-01
```

### 4. Save chart & model to disk
```bash
python stock_predictor.py --ticker TSLA --no-plot --save-model --output-dir results/
```

### All CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `GOOG` | Yahoo Finance ticker |
| `--train-start` | `2014-01-01` | Training period start |
| `--train-end` | `2023-01-01` | Training period end |
| `--epochs` | `25` | Training epochs |
| `--window` | `60` | Look-back window (days) |
| `--no-plot` | `False` | Save PNG instead of showing chart |
| `--save-model` | `False` | Save `.keras` model to disk |
| `--output-dir` | `.` | Output directory |

---

## 📊 Sample Output

```
──────────────────────────────────────────────────
  GOOG next-day forecast (2024-08-14): $179.43
  Last close:                          $176.85
  Expected move: ▲ $2.58 (+1.46%)
──────────────────────────────────────────────────
```

Evaluation metrics logged after each run:

```
MAE          4.8201
RMSE         6.1043
MAPE (%)     2.91%
R²           0.9412
```

---

## 🔬 How It Works

1. **Data ingestion** — `yfinance` downloads adjusted daily closes; a helper handles both flat and MultiIndex column formats.
2. **Preprocessing** — Prices are normalised to [0, 1] with `MinMaxScaler`. A sliding window of 60 days is used to create supervised learning samples `(X_t-60…t-1 → y_t)`.
3. **Training** — Three stacked LSTM layers with Dropout regularisation are trained end-to-end using the Adam optimiser and MSE loss.
4. **Evaluation** — Predictions on the unseen test period are inverse-transformed back to USD and scored with MAE, RMSE, MAPE, and R².
5. **Inference** — The last 60 days of available data are fed into the model to produce a next-day forecast.

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**.  
Predicted prices **should not** be used to make real financial decisions.  
Past price patterns do not guarantee future performance.

---

## 🛠️ Tech Stack

`Python 3.10+` · `TensorFlow 2.x / Keras` · `scikit-learn` · `yfinance` · `pandas` · `NumPy` · `Matplotlib`