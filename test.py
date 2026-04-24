"""
smoke_test.py
=============
Improved LSTM smoke test targeting better MAE via:
  - Multi-feature input  (Close, Volume, daily return, 5-day MA)
  - Longer training window & more data
  - Deeper model with BatchNormalization
  - Learning-rate schedule + early stopping
  - Walk-forward (anchored) cross-validation for honest MAE
"""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
keras.utils.set_random_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
TICKER      = "GOOG"
TRAIN_START = dt.datetime(2020, 1, 1)   # more history → richer patterns
TRAIN_END   = dt.datetime(2024, 1, 1)
TEST_END    = dt.datetime(2024, 7, 1)
WINDOW      = 60                         # 60-day look-back (was 30)
N_FEATURES  = 4                          # Close, Volume, Return, MA5
EPOCHS      = 40
BATCH_SIZE  = 32


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 4-column feature DataFrame from raw OHLCV data.

    Features
    --------
    close      – raw close price (kept for inverse-transform reference)
    volume     – normalised trading volume
    return_1d  – daily log return  (reduces non-stationarity)
    ma5        – 5-day moving average of close (short-term trend)
    """
    out = pd.DataFrame(index=df.index)
    close = _get_close(df)
    out["close"]     = close
    out["volume"]    = df["Volume"].squeeze()
    out["return_1d"] = np.log(close / close.shift(1))
    out["ma5"]       = close.rolling(5).mean()
    return out.dropna()


def _get_close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df.xs("Close", axis=1, level=0).iloc[:, 0]
    return df["Close"]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder  (multi-feature)
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(scaled: np.ndarray, window: int, n_features: int):
    """
    Slide a window over *scaled* to produce (X, y) pairs.

    X shape : (samples, window, n_features)
    y shape : (samples,)   — target is the *first* feature (scaled close)
    """
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window : i, :])   # all features in window
        y.append(scaled[i, 0])                 # predict Close only
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# Model  — deeper + regularised
# ─────────────────────────────────────────────────────────────────────────────

def build_model(window: int, n_features: int) -> keras.Model:
    """
    Stacked LSTM with BatchNorm and L2 regularisation.

    BatchNormalization after each LSTM layer dramatically stabilises
    training on short runs and consistently lowers validation loss.
    """
    reg = keras.regularizers.l2(1e-4)

    model = keras.models.Sequential([
        keras.layers.Input(shape=(window, n_features)),

        keras.layers.LSTM(128, return_sequences=True,
                          kernel_regularizer=reg, recurrent_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.LSTM(64, return_sequences=True,
                          kernel_regularizer=reg, recurrent_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.LSTM(32, kernel_regularizer=reg),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),

        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ], name="ImprovedLSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",                   # Huber loss — more robust to price spikes
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation  (anchored split)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_mae(
    features: np.ndarray,
    close_prices: np.ndarray,
    scaler: MinMaxScaler,
    window: int,
    n_features: int,
    n_splits: int = 3,
) -> float:
    """
    Anchored walk-forward CV: training set always starts from index 0
    and grows with each fold. Returns mean MAE across folds.

    This gives a much more honest estimate than a single train/test split.
    """
    total   = len(features)
    fold_sz = total // (n_splits + 1)
    maes    = []

    for k in range(1, n_splits + 1):
        train_end_idx = fold_sz * k
        test_end_idx  = min(fold_sz * (k + 1), total)

        train_raw = features[:train_end_idx]
        test_raw  = features[train_end_idx : test_end_idx]
        if len(test_raw) <= window:
            continue

        # Fit scaler on train only
        fold_scaler = MinMaxScaler()
        scaled_train = fold_scaler.fit_transform(train_raw)

        # Scale test using train statistics
        combined_raw    = np.vstack([train_raw, test_raw])
        scaled_combined = fold_scaler.transform(combined_raw)
        scaled_test_inp = scaled_combined[-(len(test_raw) + window):]

        X_tr, y_tr = build_sequences(scaled_train, window, n_features)
        X_te, _    = build_sequences(scaled_test_inp, window, n_features)

        m = build_model(window, n_features)
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                          monitor="loss"),
        ]
        m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, callbacks=callbacks)

        preds_scaled = m.predict(X_te, verbose=0)

        # Inverse-transform: reconstruct full-feature rows, replace col-0
        dummy         = np.zeros((len(preds_scaled), n_features))
        dummy[:, 0]   = preds_scaled[:, 0]
        preds_price   = fold_scaler.inverse_transform(dummy)[:, 0]

        actual = close_prices[train_end_idx : train_end_idx + len(preds_price)]
        mae    = mean_absolute_error(actual, preds_price)
        maes.append(mae)
        print(f"  Fold {k}  |  samples train={len(X_tr)}  test={len(X_te)}"
              f"  |  MAE=${mae:.4f}")

    return float(np.mean(maes)) if maes else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test() -> None:
    print("─" * 60)
    print(f"  LSTM Smoke Test  |  {TICKER}  |  window={WINDOW}  features={N_FEATURES}")
    print("─" * 60)

    # ── Download ──────────────────────────────────────────────────────────────
    raw_train = yf.download(TICKER, start=TRAIN_START, end=TRAIN_END,
                            progress=False, auto_adjust=True)
    raw_test  = yf.download(TICKER, start=TRAIN_END,  end=TEST_END,
                            progress=False, auto_adjust=True)

    feat_train = engineer_features(raw_train)
    feat_test  = engineer_features(raw_test)

    close_train = feat_train["close"].values
    close_test  = feat_test["close"].values

    # ── Scale (fit on train only) ─────────────────────────────────────────────
    scaler       = MinMaxScaler()
    scaled_train = scaler.fit_transform(feat_train.values)

    feat_combined   = np.vstack([feat_train.values, feat_test.values])
    scaled_combined = scaler.transform(feat_combined)
    scaled_test_inp = scaled_combined[-(len(feat_test) + WINDOW):]

    # ── Build sequences ───────────────────────────────────────────────────────
    X_train, y_train = build_sequences(scaled_train, WINDOW, N_FEATURES)
    X_test,  _       = build_sequences(scaled_test_inp, WINDOW, N_FEATURES)

    assert X_train.ndim == 3 and X_train.shape[2] == N_FEATURES, \
        f"Expected shape (N,{WINDOW},{N_FEATURES}), got {X_train.shape}"

    print(f"  Train sequences : {X_train.shape[0]}")
    print(f"  Test  sequences : {X_test.shape[0]}")

    # ── Train final model ─────────────────────────────────────────────────────
    model = build_model(WINDOW, N_FEATURES)
    model.summary(print_fn=lambda s: None)          # suppress verbose summary

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                      monitor="loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5,
                                          patience=4, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on hold-out test set ────────────────────────────────────────
    preds_scaled = model.predict(X_test, verbose=0)

    dummy       = np.zeros((len(preds_scaled), N_FEATURES))
    dummy[:, 0] = preds_scaled[:, 0]
    preds       = scaler.inverse_transform(dummy)[:, 0]

    actual = close_test[: len(preds)]
    mae    = mean_absolute_error(actual, preds)
    rmse   = np.sqrt(mean_squared_error(actual, preds))
    mape   = np.mean(np.abs((actual - preds) / actual)) * 100
    r2     = r2_score(actual, preds)

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    all_features  = np.vstack([feat_train.values, feat_test.values])
    all_close     = np.concatenate([close_train, close_test])
    print("\n  Walk-forward cross-validation (3 folds):")
    cv_mae = walk_forward_mae(all_features, all_close, scaler,
                               WINDOW, N_FEATURES, n_splits=3)

    # ── Next-day forecast ─────────────────────────────────────────────────────
    last_seq        = scaled_test_inp[-WINDOW:].reshape(1, WINDOW, N_FEATURES)
    nxt_scaled      = model.predict(last_seq, verbose=0)
    dummy_nxt       = np.zeros((1, N_FEATURES))
    dummy_nxt[0, 0] = nxt_scaled[0, 0]
    next_day_price  = scaler.inverse_transform(dummy_nxt)[0, 0]

    # ── Sanity assertions ─────────────────────────────────────────────────────
    assert np.isfinite(mae),            "MAE is not finite"
    assert np.isfinite(next_day_price), "Next-day prediction is not finite"

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  ✅ Smoke test passed")
    print("─" * 60)
    print(f"  Hold-out MAE    : ${mae:.4f}")
    print(f"  Hold-out RMSE   : ${rmse:.4f}")
    print(f"  Hold-out MAPE   : {mape:.2f}%")
    print(f"  Hold-out R²     : {r2:.4f}")
    print(f"  Walk-forward MAE: ${cv_mae:.4f}  (mean over 3 folds)")
    print(f"  Epochs trained  : {len(history.history['loss'])}  "
          f"(early stop at {EPOCHS} max)")
    print(f"  Next-day price  : ${next_day_price:.2f}")
    print("─" * 60)


if __name__ == "__main__":
    run_smoke_test()