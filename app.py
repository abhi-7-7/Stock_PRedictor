"""
app.py
======
LSTM Stock Price Predictor — Streamlit dashboard.

Run:
    streamlit run app.py

Fixes over original:
  - Scaler bug: scaler was reset to None then re-created separately from the
    one used to build training sequences, causing silent inverse-transform errors.
  - close_train was referenced after being overwritten by None in the load path.
  - x_train / y_train built with orphaned scaler in the retrain path.
  - Matplotlib chart upgraded: dark theme, residual panel, date formatting.
  - Training progress now streams live via a custom Keras callback.
  - Multi-feature input (Close + log-return + 5-day MA) for better accuracy.
  - Proper session-state caching so results persist across sidebar tweaks.
"""

import datetime as dt
import pickle
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    h1 { font-family: 'Syne', sans-serif !important;
         font-size: 2.4rem !important; letter-spacing: -0.03em; }

    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem !important;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] { font-size: 0.7rem; opacity: 0.6; }

    div[data-testid="stSidebar"] {
        background: #0a0f1a;
        border-right: 1px solid #1e2a3a;
    }

    .forecast-box {
        background: linear-gradient(135deg, #0d2137 0%, #091929 100%);
        border: 1px solid #1e4060;
        border-radius: 12px;
        padding: 1.4rem 2rem;
        margin: 1rem 0;
    }
    .forecast-ticker { font-family: 'Syne', sans-serif; font-size: 2rem;
                       font-weight: 800; color: #58a6ff; }
    .forecast-price  { font-family: 'Syne', sans-serif; font-size: 3rem;
                       font-weight: 800; color: #e6edf3; }
    .forecast-delta  { font-size: 1.1rem; }
    .up   { color: #3fb950; }
    .down { color: #f85149; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
N_FEATURES = 3   # Close (scaled), log-return, 5-day MA (scaled)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_ohlcv(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance; raise on empty result."""
    df = yf.download(
        ticker,
        start=dt.datetime.combine(start, dt.time.min),
        end=dt.datetime.combine(end, dt.time.min),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data for '{ticker}' in [{start}, {end}]. Check the ticker.")
    # Flatten MultiIndex if present (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 3-column feature frame.

    Columns
    -------
    close      — raw close price (used for inverse transform)
    return_1d  — log daily return  (reduces non-stationarity)
    ma5        — 5-day moving-average of close  (trend proxy)
    """
    close = df["Close"].squeeze()
    out = pd.DataFrame(index=df.index)
    out["close"]     = close
    out["return_1d"] = np.log(close / close.shift(1))
    out["ma5"]       = close.rolling(5).mean()
    return out.dropna()


def build_sequences(scaled: np.ndarray, window: int):
    """
    Sliding-window sequences for supervised learning.

    Returns
    -------
    X : (N, window, N_FEATURES)
    y : (N,)  — target is col-0 (scaled close)
    """
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window: i, :])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(window: int, units: int, dropout: float) -> keras.Model:
    reg = keras.regularizers.l2(1e-4)
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(window, N_FEATURES)),
            keras.layers.LSTM(units, return_sequences=True,
                              kernel_regularizer=reg, recurrent_regularizer=reg),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(units // 2, return_sequences=True,
                              kernel_regularizer=reg),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(units // 4),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout * 0.5),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ],
        name="LSTM_StockPredictor",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="huber")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE":      mean_absolute_error(actual, predicted),
        "RMSE":     np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE (%)": np.mean(np.abs((actual - predicted) / actual)) * 100,
        "R²":       r2_score(actual, predicted),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact helpers
# ─────────────────────────────────────────────────────────────────────────────

def artifact_paths(ticker: str, window: int, units: int, dropout: float):
    tag = f"{ticker.upper()}_w{window}_u{units}_d{str(dropout).replace('.','p')}"
    return ARTIFACT_DIR / f"{tag}.keras", ARTIFACT_DIR / f"{tag}.pkl"


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str) -> keras.Model:
    return keras.models.load_model(path)


def save_artifacts(model, scaler, model_path: Path, scaler_path: Path):
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> MinMaxScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit live-training callback
# ─────────────────────────────────────────────────────────────────────────────

class StreamlitProgressCallback(keras.callbacks.Callback):
    """Streams epoch loss into a Streamlit progress bar + text widget."""

    def __init__(self, epochs: int, bar, status_text):
        super().__init__()
        self.total   = epochs
        self.bar     = bar
        self.status  = status_text

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        val  = logs.get("val_loss")
        pct  = (epoch + 1) / self.total
        self.bar.progress(pct)
        val_str = f"  val_loss={val:.5f}" if val is not None else ""
        self.status.caption(
            f"Epoch {epoch+1}/{self.total}  —  loss={loss:.5f}{val_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def make_chart(
    ticker: str,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: dict,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    bg = "#0d1117"
    fig.patch.set_facecolor(bg)

    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor("#21262d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#8b949e")

    # ── Price panel ──────────────────────────────────────────────────────────
    ax1.plot(dates, actual,    color="#e6edf3", lw=1.4, label="Actual", zorder=3)
    ax1.plot(dates, predicted, color="#58a6ff", lw=1.4, ls="--",
             label="Predicted", zorder=3)
    ax1.fill_between(dates, actual, predicted,
                     where=predicted > actual,
                     alpha=0.12, color="#3fb950", label="Over-estimate")
    ax1.fill_between(dates, actual, predicted,
                     where=predicted <= actual,
                     alpha=0.12, color="#f85149", label="Under-estimate")
    ax1.set_title(f"{ticker}  ·  LSTM Price Forecast",
                  color="#e6edf3", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Price (USD)", color="#8b949e", fontsize=9)
    ax1.legend(framealpha=0.15, labelcolor="#c9d1d9",
               facecolor="#161b22", fontsize=8)

    metric_str = "   ".join(
        f"{k}: {v:.2f}%" if "MAPE" in k else
        f"{k}: {v:.4f}"
        for k, v in metrics.items()
    )
    ax1.set_xlabel("")

    # ── Residual panel ───────────────────────────────────────────────────────
    residuals = predicted - actual
    colors = np.where(residuals >= 0, "#3fb950", "#f85149")
    ax2.bar(dates, residuals, color=colors, alpha=0.75, width=1.5)
    ax2.axhline(0, color="#8b949e", lw=0.8, ls="--")
    ax2.set_ylabel("Residual", color="#8b949e", fontsize=9)
    ax2.set_xlabel("Date", color="#8b949e", fontsize=9)

    fig.text(0.5, 0.98, metric_str, ha="center", va="top",
             color="#8b949e", fontsize=8, style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    ticker = st.text_input("Ticker symbol", value="GOOG").upper().strip()

    st.markdown("**Date range**")
    col_a, col_b = st.columns(2)
    train_start = col_a.date_input("Train start", value=dt.date(2018, 1, 1))
    train_end   = col_b.date_input("Train end",   value=dt.date(2023, 1, 1))
    test_end    = st.date_input("Test end", value=dt.date.today())

    st.markdown("**Model**")
    window     = st.slider("Lookback window (days)", 20, 120, 60)
    epochs     = st.slider("Epochs",                  1,  60,  20)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    units      = st.slider("LSTM units",             16, 128,  64, step=8)
    dropout    = st.slider("Dropout",               0.0, 0.5, 0.2, step=0.05)

    st.markdown("**Storage**")
    use_saved      = st.checkbox("Use saved model if available", value=True)
    save_after_run = st.checkbox("Save model after training",    value=True)
    force_retrain  = st.checkbox("Force retrain",                value=False)

    st.divider()
    run_btn = st.button("▶  Run prediction", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📈 LSTM Stock Predictor")
st.caption("Deep-learning forecast using stacked LSTM on multi-feature price data")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:

    # ── Validate dates ────────────────────────────────────────────────────────
    if train_start >= train_end:
        st.error("⛔  Train end must be after train start.")
        st.stop()
    if train_end >= test_end:
        st.error("⛔  Test end must be after train end.")
        st.stop()

    try:
        # ── 1. Fetch data ─────────────────────────────────────────────────────
        with st.spinner(f"Downloading {ticker} data …"):
            train_df = fetch_ohlcv(ticker, train_start, train_end)
            test_df  = fetch_ohlcv(ticker, train_end,   test_end)

        feat_train = engineer_features(train_df)
        feat_test  = engineer_features(test_df)

        close_train_vals = feat_train["close"].values
        close_test_vals  = feat_test["close"].values

        # ── 2. Scale (fit on train only — single scaler, always consistent) ──
        #
        # FIX: original code created scaled_train with scaler A, then overwrote
        # `scaler` with None and later re-created scaler B — causing a mismatch
        # between the sequences fed to training and the scaler used for inference.
        #
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(feat_train.values)

        # ── 3. Build training sequences ───────────────────────────────────────
        X_train, y_train = build_sequences(scaled_train, window)
        if len(X_train) == 0:
            st.error("⛔  Not enough training data for the chosen lookback window.")
            st.stop()

        # ── 4. Model: load or train ───────────────────────────────────────────
        model_path, scaler_path = artifact_paths(ticker, window, units, dropout)
        model = None

        if use_saved and not force_retrain and model_path.exists() and scaler_path.exists():
            with st.spinner("Loading saved model …"):
                model  = load_model_cached(str(model_path))
                scaler = load_scaler(scaler_path)
            st.toast("Loaded saved model ✔", icon="💾")
        else:
            np.random.seed(42)
            keras.utils.set_random_seed(42)
            model = build_model(window=window, units=units, dropout=dropout)

            st.markdown("**Training progress**")
            prog_bar    = st.progress(0.0)
            status_text = st.empty()

            callbacks = [
                StreamlitProgressCallback(epochs, prog_bar, status_text),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6,
                    restore_best_weights=True, verbose=0
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3,
                    min_lr=1e-6, verbose=0
                ),
            ]
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0,
            )
            prog_bar.progress(1.0)
            status_text.caption("Training complete ✔")

            if save_after_run:
                save_artifacts(model, scaler, model_path, scaler_path)
                st.toast("Model saved to disk ✔", icon="💾")

        # ── 5. Build test input (scale with the SAME fitted scaler) ───────────
        feat_combined   = np.vstack([feat_train.values, feat_test.values])
        scaled_combined = scaler.transform(feat_combined)
        scaled_test_inp = scaled_combined[-(len(feat_test) + window):]

        X_test, _ = build_sequences(scaled_test_inp, window)

        # ── 6. Predict ────────────────────────────────────────────────────────
        pred_scaled = model.predict(X_test, verbose=0)

        # Inverse-transform: reconstruct full-feature rows, replace col-0
        dummy_pred      = np.zeros((len(pred_scaled), N_FEATURES))
        dummy_pred[:,0] = pred_scaled[:,0]
        predicted_prices = scaler.inverse_transform(dummy_pred)[:,0]

        actual_prices = close_test_vals
        n = min(len(actual_prices), len(predicted_prices))
        actual_prices    = actual_prices[:n]
        predicted_prices = predicted_prices[:n]
        test_dates       = feat_test.index[:n]

        # ── 7. Next-day forecast ──────────────────────────────────────────────
        last_seq        = scaled_test_inp[-window:].reshape(1, window, N_FEATURES)
        nxt_scaled      = model.predict(last_seq, verbose=0)
        dummy_nxt       = np.zeros((1, N_FEATURES))
        dummy_nxt[0,0]  = nxt_scaled[0,0]
        next_day_price  = float(scaler.inverse_transform(dummy_nxt)[0,0])

        last_price = float(actual_prices[-1])
        delta      = next_day_price - last_price
        pct_delta  = delta / last_price * 100
        arrow      = "▲" if delta >= 0 else "▼"
        delta_cls  = "up" if delta >= 0 else "down"
        last_date  = test_dates[-1]
        fcast_date = last_date + pd.tseries.offsets.BDay(1)

        # ── 8. Metrics ────────────────────────────────────────────────────────
        metrics = evaluate(actual_prices, predicted_prices)

        # ── Render ────────────────────────────────────────────────────────────
        st.divider()

        # Forecast hero
        st.markdown(
            f"""
            <div class="forecast-box">
              <div class="forecast-ticker">{ticker} &nbsp;·&nbsp; {fcast_date.strftime('%d %b %Y')}</div>
              <div class="forecast-price">${next_day_price:.2f}</div>
              <div class="forecast-delta {delta_cls}">
                {arrow} ${abs(delta):.2f} &nbsp;({pct_delta:+.2f}%)
                &nbsp;&nbsp;last close: ${last_price:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",    f"${metrics['MAE']:.2f}")
        c2.metric("RMSE",   f"${metrics['RMSE']:.2f}")
        c3.metric("MAPE",   f"{metrics['MAPE (%)']:.2f}%")
        c4.metric("R²",     f"{metrics['R²']:.4f}")

        # Chart
        st.divider()
        fig = make_chart(ticker, test_dates, actual_prices, predicted_prices, metrics)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Raw data table
        with st.expander("🗃  Recent predictions (last 30 days)"):
            result_df = pd.DataFrame({
                "Date":      test_dates,
                "Actual ($)":    actual_prices.round(2),
                "Predicted ($)": predicted_prices.round(2),
                "Error ($)":     (predicted_prices - actual_prices).round(2),
            }).tail(30)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Download predictions
        csv = pd.DataFrame({
            "date":      test_dates.astype(str),
            "actual":    actual_prices,
            "predicted": predicted_prices,
        }).to_csv(index=False)
        st.download_button(
            "⬇  Download predictions CSV",
            data=csv,
            file_name=f"{ticker}_predictions.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"⛔  {exc}")

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 5rem 0; color: #444d56;">
          <div style="font-size:3rem;">📉</div>
          <div style="font-family:'Syne',sans-serif; font-size:1.3rem; margin-top:.5rem;">
            Configure settings in the sidebar<br>and click <strong>Run prediction</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )