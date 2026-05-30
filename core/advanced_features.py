import numpy as np
import pandas as pd

from core.feature_engine import FeatureEngine, FEATURE_COLS
from utils.logger import get_logger

logger = get_logger("advanced_features")

ADVANCED_FEATURE_COLS: list[str] = [
    "vwap",
    "vwap_dist",
    "chop",
    "fisher",
    "orderflow",
    "orderflow_ma",
    "session_asian",
    "session_eu",
    "session_ny",
    "session_overlap",
    "ha_bull",
    "ha_bear",
    "ha_body",
    "higher_high",
    "lower_low",
    "inside_bar",
    "vol_ratio_50",
    "trend_strength",
    "rsi_slope",
    "price_slope",
    "rsi_div",
    "squeeze",
    "cum_delta",
    "rsi_lag1",
    "close_lag1",
    "rsi_lag2",
    "close_lag2",
    "rsi_lag3",
    "close_lag3",
]


def _heikin_ashi(df: pd.DataFrame):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


class AdvancedFeatureEngine(FeatureEngine):
    def compute_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute(df)
        df = self._add_vwap(df)
        df = self._add_chop(df)
        df = self._add_fisher(df)
        df = self._add_orderflow(df)
        df = self._add_sessions(df)
        df = self._add_heikin_ashi(df)
        df = self._add_price_action(df)
        df = self._add_vol_ratio_50(df)
        df = self._add_trend_strength(df)
        df = self._add_rsi_divergence(df)
        df = self._add_squeeze(df)
        df = self._add_cum_delta(df)
        df = self._add_lags(df)

        logger.debug(
            "AdvancedFeatureEngine.compute_advanced: %d rows × %d total feature cols",
            len(df),
            len(FEATURE_COLS) + len(ADVANCED_FEATURE_COLS),
        )
        return df

    def get_all_feature_cols(self) -> list[str]:
        return list(FEATURE_COLS) + list(ADVANCED_FEATURE_COLS)

    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        vol = df["tick_volume"].fillna(1).replace(0, 1)
        df["vwap"] = (vol * df["close"]).cumsum() / vol.cumsum()
        df["vwap_dist"] = (df["close"] - df["vwap"]) / (df["close"] + 1e-9)
        return df

    @staticmethod
    def _add_chop(df: pd.DataFrame) -> pd.DataFrame:
        n = 14
        atr1 = df["high"] - df["low"]
        df["chop"] = (
            100
            * np.log10(
                atr1.rolling(n).sum()
                / (df["high"].rolling(n).max() - df["low"].rolling(n).min() + 1e-9)
            )
            / np.log10(n)
        )
        return df

    @staticmethod
    def _add_fisher(df: pd.DataFrame) -> pd.DataFrame:
        median = (df["high"] + df["low"]) / 2
        highest = median.rolling(9).max()
        lowest = median.rolling(9).min()
        value = 2 * (median - lowest) / (highest - lowest + 1e-9) - 1
        value = value.clip(-0.999, 0.999)
        df["fisher"] = 0.5 * np.log((1 + value) / (1 - value + 1e-9))
        return df

    @staticmethod
    def _add_orderflow(df: pd.DataFrame) -> pd.DataFrame:
        spread = df["spread"] if "spread" in df.columns else pd.Series(1.0, index=df.index)
        df["orderflow"] = (
            df["tick_volume"] * np.sign(df["close"] - df["open"]) / (spread + 1e-9)
        )
        df["orderflow_ma"] = df["orderflow"].rolling(10).mean()
        return df

    @staticmethod
    def _add_sessions(df: pd.DataFrame) -> pd.DataFrame:
        hour = pd.to_datetime(df["time"]).dt.hour
        df["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)
        df["session_eu"] = ((hour >= 7) & (hour < 16)).astype(float)
        df["session_ny"] = ((hour >= 13) & (hour < 22)).astype(float)
        df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)
        return df

    @staticmethod
    def _add_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
        ha_o, ha_h, ha_l, ha_c = _heikin_ashi(df)
        df["ha_bull"] = (
            (ha_c > ha_o) & (ha_c.shift(1) > ha_o.shift(1))
        ).astype(float)
        df["ha_bear"] = (
            (ha_c < ha_o) & (ha_c.shift(1) < ha_o.shift(1))
        ).astype(float)
        df["ha_body"] = (ha_c - ha_o).abs() / (ha_h - ha_l + 1e-9)
        return df

    @staticmethod
    def _add_price_action(df: pd.DataFrame) -> pd.DataFrame:
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(float)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(float)
        df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
        ).astype(float)
        return df

    @staticmethod
    def _add_vol_ratio_50(df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ratio_50"] = df["atr"] / (df["atr"].shift(50) + 1e-9)
        return df

    @staticmethod
    def _add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
        df["trend_strength"] = (
            sum(
                (df["close"].shift(i) > df["ema21"].shift(i)).astype(float)
                for i in range(5)
            )
            / 5
        )
        return df

    @staticmethod
    def _add_rsi_divergence(df: pd.DataFrame) -> pd.DataFrame:
        df["rsi_slope"] = df["rsi"] - df["rsi"].shift(3)
        df["price_slope"] = (df["close"] - df["close"].shift(3)) / (
            df["close"].shift(3) + 1e-9
        )
        df["rsi_div"] = np.sign(df["rsi_slope"]) - np.sign(df["price_slope"])
        return df

    @staticmethod
    def _add_squeeze(df: pd.DataFrame) -> pd.DataFrame:
        kc_upper = df["ema21"] + 1.5 * df["atr"]
        kc_lower = df["ema21"] - 1.5 * df["atr"]
        bb_width = (
            df["bb_upper"] - df["bb_lower"]
            if "bb_upper" in df.columns
            else df["atr"] * 4
        )
        kc_width = kc_upper - kc_lower
        df["squeeze"] = (bb_width < kc_width).astype(float)
        return df

    @staticmethod
    def _add_cum_delta(df: pd.DataFrame) -> pd.DataFrame:
        df["cum_delta"] = (
            df["tick_volume"] * np.sign(df["close"] - df["open"])
        ).rolling(20).sum()
        return df

    @staticmethod
    def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 2, 3]:
            df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
            df[f"close_lag{lag}"] = df["close"].pct_change(lag)
        return df
