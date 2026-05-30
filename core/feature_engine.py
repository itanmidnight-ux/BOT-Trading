"""
feature_engine.py
Computes technical indicator features from OHLCV DataFrames.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from utils.logger import get_logger

logger = get_logger("feature_engine")

# ── Columns that are NOT features (raw OHLCV + derived meta) ──────────────────
_OHLCV_COLS = {"time", "open", "high", "low", "close", "tick_volume", "spread"}

# Full ordered list of feature columns produced by compute()
FEATURE_COLS: list[str] = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "bb_pct",
    "bb_width",
    "ema9",
    "ema21",
    "ema50",
    "ema_diff",
    "adx",
    "adx_pos",
    "adx_neg",
    "stoch_k",
    "stoch_d",
    "cci",
    "willr",
    "vol_ratio",
    "body_ratio",
    "price_pos",
    "momentum10",
    "hour_sin",
    "hour_cos",
    "atr_mean50",
]


class FeatureEngine:
    """
    Computes all technical features from a raw OHLCV DataFrame.

    The input DataFrame must contain at least:
        time, open, high, low, close, tick_volume

    The 'spread' column is optional and is ignored during feature computation.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all feature columns to a copy of *df* and drops the first 60 rows
        (which contain NaN values due to look-back periods of the indicators).

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data. Must contain: time, open, high, low, close,
            tick_volume.

        Returns
        -------
        pd.DataFrame
            DataFrame with all feature columns appended and NaN-head rows
            removed. Index is reset.
        """
        df = df.copy()
        self._validate_columns(df)

        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_atr(df)
        df = self._add_bbands(df)
        df = self._add_emas(df)
        df = self._add_adx(df)
        df = self._add_stochastic(df)
        df = self._add_cci(df)
        df = self._add_willr(df)
        df = self._add_volume_features(df)
        df = self._add_candle_features(df)
        df = self._add_momentum(df)
        df = self._add_time_features(df)
        df = self._add_atr_mean(df)

        # Drop the initial rows that will have NaN from long look-backs
        df = df.iloc[60:].reset_index(drop=True)

        remaining_nans = df[FEATURE_COLS].isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(
                "FeatureEngine.compute: %d NaN values remain after dropping "
                "first 60 rows. Consider passing more historical data.",
                remaining_nans,
            )

        logger.debug(
            "FeatureEngine.compute: produced %d rows × %d feature cols",
            len(df),
            len(FEATURE_COLS),
        )
        return df

    def get_latest_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the last row of the feature-enriched DataFrame as a Series.
        Useful for live trading inference.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data (will be passed through compute() internally if
            feature columns are absent; otherwise the last row is returned
            directly).

        Returns
        -------
        pd.Series
            Named series with all feature columns for the most recent candle.
        """
        missing_features = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_features:
            df = self.compute(df)

        latest = df.iloc[-1][FEATURE_COLS]
        logger.debug(
            "FeatureEngine.get_latest_features: returning features for "
            "last candle (index=%d)",
            df.index[-1],
        )
        return latest

    @staticmethod
    def get_feature_cols() -> list[str]:
        """Returns the ordered list of feature column names (no OHLCV, no target)."""
        return list(FEATURE_COLS)

    @staticmethod
    def add_target(df: pd.DataFrame, symbol: str = "EURUSD") -> pd.DataFrame:
        """
        Appends a binary classification target to *df*.

        The target is 1 when the close price 3 candles ahead exceeds the
        current close by at least *pip_th*, otherwise 0.

        Parameters
        ----------
        df     : pd.DataFrame with a 'close' column.
        symbol : Instrument name used to select the pip threshold.

        Returns
        -------
        pd.DataFrame with 'target' column added (last 3 rows will be NaN).
        """
        if "USD" in symbol and "XAU" not in symbol:
            pip_th = 0.0003
        else:
            pip_th = 0.15

        df["target"] = (
            df["close"].shift(-3) > df["close"] + pip_th
        ).astype(float)

        # The last 3 rows have no valid future close → set to NaN
        df.loc[df.index[-3:], "target"] = np.nan

        logger.debug(
            "add_target: symbol=%s pip_th=%.5f target_mean=%.4f",
            symbol,
            pip_th,
            df["target"].mean(),
        )
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "tick_volume", "time"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"FeatureEngine: missing required columns: {missing}"
            )

    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.rsi(df["close"], length=14)
        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is None or macd.empty:
            df["macd"] = np.nan
            df["macd_signal"] = np.nan
            df["macd_hist"] = np.nan
        else:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]
        return df

    @staticmethod
    def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        return df

    @staticmethod
    def _add_bbands(df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"], length=20, std=2)
        if bb is None or bb.empty:
            for col in ("bb_upper", "bb_lower", "bb_mid", "bb_pct", "bb_width"):
                df[col] = np.nan
        else:
            # pandas-ta ≥0.3.14b colnames: BBU_20_2.0_2.0 / BBU_20_2.0
            # Resolve dynamically to support both naming conventions.
            col_upper = next(c for c in bb.columns if c.startswith("BBU_"))
            col_lower = next(c for c in bb.columns if c.startswith("BBL_"))
            col_mid   = next(c for c in bb.columns if c.startswith("BBM_"))
            df["bb_upper"] = bb[col_upper]
            df["bb_lower"] = bb[col_lower]
            df["bb_mid"]   = bb[col_mid]
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"] + 1e-9
            )
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (
                df["bb_mid"] + 1e-9
            )
        return df

    @staticmethod
    def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
        df["ema9"] = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["ema_diff"] = (df["ema9"] - df["ema21"]) / (df["ema21"] + 1e-9)
        return df

    @staticmethod
    def _add_adx(df: pd.DataFrame) -> pd.DataFrame:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx is None or adx.empty:
            df["adx"] = np.nan
            df["adx_pos"] = np.nan
            df["adx_neg"] = np.nan
        else:
            df["adx"] = adx["ADX_14"]
            df["adx_pos"] = adx["DMP_14"]
            df["adx_neg"] = adx["DMN_14"]
        return df

    @staticmethod
    def _add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        if stoch is None or stoch.empty:
            df["stoch_k"] = np.nan
            df["stoch_d"] = np.nan
        else:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
        return df

    @staticmethod
    def _add_cci(df: pd.DataFrame) -> pd.DataFrame:
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
        return df

    @staticmethod
    def _add_willr(df: pd.DataFrame) -> pd.DataFrame:
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)
        return df

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ratio"] = df["tick_volume"] / (
            df["tick_volume"].rolling(20).mean() + 1e-9
        )
        return df

    @staticmethod
    def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
        df["body_ratio"] = (df["close"] - df["open"]).abs() / (
            df["high"] - df["low"] + 1e-9
        )
        df["price_pos"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-9
        )
        return df

    @staticmethod
    def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        df["momentum10"] = df["close"] - df["close"].shift(10)
        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        hours = pd.to_datetime(df["time"]).dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        return df

    @staticmethod
    def _add_atr_mean(df: pd.DataFrame) -> pd.DataFrame:
        df["atr_mean50"] = df["atr"].rolling(50).mean()
        return df
