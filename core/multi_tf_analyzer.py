"""
multi_tf_analyzer.py
Aggregates signals from multiple timeframes to derive a trading consensus.
"""

from typing import Optional

import pandas as pd

from config import constants, settings
from utils.logger import get_logger

logger = get_logger("multi_tf_analyzer")

# Signal constants
_BUY     = constants.SIGNAL_BUY    # "BUY"
_SELL    = constants.SIGNAL_SELL   # "SELL"
_NEUTRAL = constants.SIGNAL_HOLD   # "HOLD"  (used as NEUTRAL internally)

# Penalty applied when the M15 timeframe opposes the proposed direction
_M15_PENALTY = 0.08

# Minimum votes required to declare a consensus direction
_MIN_VOTES = 2

# Timeframe evaluated for the penalty
_PENALTY_TF = "M15"


class MultiTFAnalyzer:
    """
    Maintains in-memory feature DataFrames for (symbol, timeframe) pairs and
    computes a multi-timeframe signal consensus.

    Usage
    -----
    >>> analyzer = MultiTFAnalyzer()
    >>> analyzer.update("EURUSD", "M1",  df_m1)
    >>> analyzer.update("EURUSD", "M5",  df_m5)
    >>> analyzer.update("EURUSD", "M15", df_m15)
    >>> result = analyzer.get_consensus("EURUSD", "BUY")
    >>> # {'consensus': 'BUY', 'votes': 2, 'penalty': 0.0}
    """

    def __init__(self) -> None:
        # Key: (symbol, tf_str)  Value: pd.DataFrame (feature-enriched)
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, symbol: str, tf_str: str, df: pd.DataFrame) -> None:
        """
        Store or replace the feature DataFrame for a (symbol, timeframe) pair.

        Parameters
        ----------
        symbol : str   Instrument name, e.g. "EURUSD".
        tf_str : str   Timeframe string, e.g. "M1", "M5", "M15".
        df     : pd.DataFrame  Feature-enriched DataFrame from FeatureEngine.
        """
        if df is None or df.empty:
            logger.warning(
                "MultiTFAnalyzer.update: received empty DataFrame for "
                "(%s, %s) — skipping",
                symbol, tf_str,
            )
            return

        key = (symbol.upper(), tf_str.upper())
        self._frames[key] = df
        logger.debug(
            "MultiTFAnalyzer.update: stored %d rows for (%s, %s)",
            len(df), symbol, tf_str,
        )

    def get_consensus(
        self,
        symbol: str,
        direction: str,
    ) -> dict:
        """
        Evaluate multi-timeframe agreement for *direction* on *symbol*.

        Parameters
        ----------
        symbol    : str   Instrument name, e.g. "EURUSD".
        direction : str   Proposed trade direction: "BUY" or "SELL".

        Returns
        -------
        dict with keys:
            consensus : "BUY" | "SELL" | "NEUTRAL"
            votes     : int — number of timeframes that agree with *direction*
            penalty   : float — 0.08 if M15 opposes *direction*, else 0.0
        """
        symbol = symbol.upper()
        direction = direction.upper()

        available_tfs = [
            tf for (sym, tf) in self._frames if sym == symbol
        ]

        if not available_tfs:
            logger.warning(
                "MultiTFAnalyzer.get_consensus: no data for symbol=%s",
                symbol,
            )
            return {"consensus": _NEUTRAL, "votes": 0, "penalty": 0.0}

        bullish_votes = 0
        bearish_votes = 0
        tf_signals: dict[str, str] = {}

        for tf in available_tfs:
            key = (symbol, tf)
            df  = self._frames[key]
            sig = self._classify_tf(df, tf)
            tf_signals[tf] = sig
            if sig == _BUY:
                bullish_votes += 1
            elif sig == _SELL:
                bearish_votes += 1

        # ── Determine consensus ───────────────────────────────────────────────
        if bullish_votes >= _MIN_VOTES:
            consensus = _BUY
        elif bearish_votes >= _MIN_VOTES:
            consensus = _SELL
        else:
            consensus = _NEUTRAL

        # ── M15 penalty ───────────────────────────────────────────────────────
        penalty = 0.0
        m15_sig = tf_signals.get(_PENALTY_TF.upper())

        if direction == _BUY and m15_sig == _SELL:
            penalty = _M15_PENALTY
            logger.debug(
                "MultiTFAnalyzer: M15 bearish while direction=BUY → "
                "penalty=%.2f",
                penalty,
            )
        elif direction == _SELL and m15_sig == _BUY:
            penalty = _M15_PENALTY
            logger.debug(
                "MultiTFAnalyzer: M15 bullish while direction=SELL → "
                "penalty=%.2f",
                penalty,
            )

        votes = bullish_votes if direction == _BUY else bearish_votes

        logger.info(
            "MultiTFAnalyzer.get_consensus: symbol=%s direction=%s "
            "consensus=%s votes=%d penalty=%.2f tf_signals=%s",
            symbol, direction, consensus, votes, penalty, tf_signals,
        )

        return {
            "consensus": consensus,
            "votes":     votes,
            "penalty":   penalty,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _classify_tf(df: pd.DataFrame, tf: str) -> str:
        """
        Classify the latest candle of a single timeframe as BUY, SELL, or
        NEUTRAL based on EMA cross, RSI filter, and MACD histogram.

        Bullish:  ema9 > ema21  AND  rsi < 70  AND  macd_hist > 0
        Bearish:  ema9 < ema21  AND  rsi > 30  AND  macd_hist < 0
        """
        if df.empty:
            return _NEUTRAL

        row = df.iloc[-1]

        ema9      = MultiTFAnalyzer._safe(row, "ema9")
        ema21     = MultiTFAnalyzer._safe(row, "ema21")
        rsi       = MultiTFAnalyzer._safe(row, "rsi")
        macd_hist = MultiTFAnalyzer._safe(row, "macd_hist")

        if any(v is None for v in (ema9, ema21, rsi, macd_hist)):
            logger.debug(
                "MultiTFAnalyzer._classify_tf: missing values on tf=%s → NEUTRAL",
                tf,
            )
            return _NEUTRAL

        is_bullish = ema9 > ema21 and rsi < 70 and macd_hist > 0
        is_bearish = ema9 < ema21 and rsi > 30 and macd_hist < 0

        if is_bullish:
            return _BUY
        if is_bearish:
            return _SELL
        return _NEUTRAL

    @staticmethod
    def _safe(row: pd.Series, col: str) -> Optional[float]:
        """Returns scalar float for *col* or None if absent / NaN."""
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        return float(val)
