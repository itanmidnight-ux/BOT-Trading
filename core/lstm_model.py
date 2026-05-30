"""
lstm_model.py

Sequence-based predictor that acts as the 4th voter in the ensemble.

Architecture
------------
Despite the class name ("LSTM"), the implementation uses an sklearn
MLPClassifier whose input is a flattened sliding window of the last
``SEQ_LEN`` timesteps.  This gives equivalent sequence-awareness while
avoiding heavy deep-learning dependencies (PyTorch / TensorFlow).

Input  : last 20 candles × 6 features = 120-dimensional feature vector
Hidden : [128, 64] with ReLU activations + early stopping
Output : binary classification  (1 = buy opportunity, 0 = no signal)

The model is serialised with joblib and stored under ``settings.MODELS_DIR``
as ``{symbol}_lstm.pkl``.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.neural_network import MLPClassifier

from utils.logger import get_logger
from config import settings

_log = get_logger("lstm_model")


class LSTMModel:
    """
    Sliding-window MLP that behaves as a sequence predictor.

    Typical usage
    -------------
    ::

        model = LSTMModel.load("EURUSD")   # or LSTMModel() if first run

        # Training
        success = model.train(df_features, y_labels)

        # Inference (pass the last SEQ_LEN + some buffer rows)
        proba = model.predict_proba(df_window)   # float in [0, 1]

        # Persist
        model.save("EURUSD")
    """

    SEQ_LEN = 20

    # Features produced by FeatureEngine that are consumed by this model.
    # ``close_pct``, ``atr_norm``, and ``macd_hist_norm`` are derived below;
    # ``rsi``, ``ema_diff``, and ``vol_ratio`` come directly from FeatureEngine.
    BASE_FEATURES = [
        "close_pct",
        "rsi",
        "atr_norm",
        "macd_hist_norm",
        "ema_diff",
        "vol_ratio",
    ]

    def __init__(self) -> None:
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        self._trained: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, y: np.ndarray) -> bool:
        """
        Build feature sequences and fit the MLP.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-enriched DataFrame produced by FeatureEngine (must contain
            at least ``close``, ``atr``; ``rsi``, ``ema_diff``, ``vol_ratio``
            used when present).
        y  : array-like of int/float, length == len(df)
            Binary labels aligned to *df* rows (1 = profitable trade, 0 = not).

        Returns
        -------
        bool  True on success, False when there are not enough samples or the
              fit raises an exception.
        """
        X = self._make_sequences(df)
        y_arr = np.asarray(y)

        # Labels must be aligned to the *output* rows (SEQ_LEN offset applied)
        y_aligned = y_arr[self.SEQ_LEN:]

        if len(X) < 100:
            _log.warning(
                "LSTM train skipped: only %d sequences (need ≥ 100)", len(X)
            )
            return False

        if len(X) != len(y_aligned):
            _log.warning(
                "LSTM train: X/y length mismatch (%d vs %d) — truncating",
                len(X), len(y_aligned),
            )
            min_len   = min(len(X), len(y_aligned))
            X         = X[:min_len]
            y_aligned = y_aligned[:min_len]

        try:
            self.model.fit(X, y_aligned)
            self._trained = True
            _log.info(
                "LSTM(MLP) trained on %d sequences (%d features each).",
                len(X), X.shape[1],
            )
            return True
        except Exception as exc:
            _log.warning("LSTM train error: %s", exc)
            return False

    def predict_proba(self, df_window: pd.DataFrame) -> float:
        """
        Predict the probability of a buy signal from the latest sequence.

        Parameters
        ----------
        df_window : pd.DataFrame
            At minimum the last ``SEQ_LEN`` rows of feature-enriched data.
            Passing a slightly larger window (e.g. SEQ_LEN + 5) is fine.

        Returns
        -------
        float  probability in [0, 1]; returns 0.5 (neutral) if the model has
               not been trained yet or if sequence construction fails.
        """
        if not self._trained:
            _log.debug("LSTM not trained — returning neutral 0.5")
            return 0.5

        X = self._make_sequences(df_window)
        if len(X) == 0:
            _log.debug("LSTM: no sequences from df_window — returning 0.5")
            return 0.5

        try:
            proba = float(self.model.predict_proba(X[-1:])[0, 1])
            _log.debug("LSTM predict_proba → %.4f", proba)
            return proba
        except Exception as exc:
            _log.warning("LSTM predict_proba error: %s", exc)
            return 0.5

    def save(self, symbol: str) -> None:
        """
        Serialise this instance to ``settings.MODELS_DIR/{symbol}_lstm.pkl``.

        Parameters
        ----------
        symbol : instrument name used as filename prefix (e.g. ``"EURUSD"``)
        """
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.MODELS_DIR / f"{symbol}_lstm.pkl"
        try:
            joblib.dump(self, path)
            _log.info("LSTM model saved → %s", path)
        except Exception as exc:
            _log.warning("LSTM save failed: %s", exc)

    @classmethod
    def load(cls, symbol: str) -> "LSTMModel":
        """
        Load a previously saved model for *symbol*, or return a fresh instance.

        Parameters
        ----------
        symbol : instrument name (e.g. ``"EURUSD"``)

        Returns
        -------
        LSTMModel  restored instance (or new untrained one if no file exists)
        """
        path = settings.MODELS_DIR / f"{symbol}_lstm.pkl"
        if path.exists():
            try:
                obj = joblib.load(path)
                if isinstance(obj, cls):
                    _log.info("LSTM model loaded from %s", path)
                    return obj
                _log.warning(
                    "Unexpected object type in %s — creating fresh model.", path
                )
            except Exception as exc:
                _log.warning("LSTM load failed (%s) — creating fresh model.", exc)
        return cls()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        Derive normalised features and build sliding-window sequences.

        Derived columns added on a copy of *df*:
        - ``close_pct``      : 1-bar close percent-change
        - ``atr_norm``       : ATR / close  (scale-free ATR)
        - ``macd_hist_norm`` : macd_hist / close  (when column exists)

        Any feature listed in ``BASE_FEATURES`` that is absent from *df*
        is silently dropped.  At least one feature must survive.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray  shape (n_windows, n_available_features × SEQ_LEN)
                    Returns empty array with shape (0, 0) on failure.
        """
        if df is None or len(df) <= self.SEQ_LEN:
            return np.empty((0, 0))

        df = df.copy()

        # ── Derived normalised features ───────────────────────────────────────
        if "close" in df.columns:
            close_safe      = df["close"].replace(0, np.nan)
            df["close_pct"] = df["close"].pct_change()
            df["atr_norm"]  = (
                df["atr"] / close_safe
                if "atr" in df.columns
                else 0.0
            )
            if "macd_hist" in df.columns:
                df["macd_hist_norm"] = df["macd_hist"] / (close_safe + 1e-9)
            else:
                df["macd_hist_norm"] = 0.0
        else:
            # Without close we can still proceed if other features exist
            df["close_pct"]      = 0.0
            df["atr_norm"]       = 0.0
            df["macd_hist_norm"] = 0.0

        # ── Select available features ─────────────────────────────────────────
        avail = [c for c in self.BASE_FEATURES if c in df.columns]
        if not avail:
            _log.warning("LSTM: no usable feature columns found in DataFrame")
            return np.empty((0, 0))

        matrix = df[avail].fillna(0).values.astype(np.float64)

        # ── Build flattened windows ───────────────────────────────────────────
        n_windows = len(matrix) - self.SEQ_LEN
        if n_windows <= 0:
            return np.empty((0, 0))

        n_feats = len(avail)
        X = np.empty((n_windows, self.SEQ_LEN * n_feats), dtype=np.float64)
        for i in range(n_windows):
            X[i] = matrix[i : i + self.SEQ_LEN].flatten()

        return X
