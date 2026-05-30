"""
regime_detector.py
Determines the current market regime from a feature-enriched DataFrame.
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import constants, settings
from utils.logger import get_logger

logger = get_logger("regime_detector")

# Convenience aliases from constants
TRENDING_UP   = constants.REGIME_TRENDING_UP
TRENDING_DOWN = constants.REGIME_TRENDING_DOWN
RANGING       = constants.REGIME_RANGING
VOLATILE      = constants.REGIME_VOLATILE
NO_TRADE      = constants.REGIME_NO_TRADE

# Regimes where the bot is allowed to open new trades
_TRADEABLE = {TRENDING_UP, TRENDING_DOWN, RANGING}

# Thresholds
_ADX_TREND_MIN  = 25.0   # ADX must exceed this to classify as trending
_ADX_RANGE_MAX  = 20.0   # ADX must be below this to classify as ranging
_ATR_VOLATILE_K = 2.5    # ATR spike multiplier vs 50-bar mean
_BB_PCT_LOW     = 0.20   # Bollinger %B lower bound for ranging
_BB_PCT_HIGH    = 0.80   # Bollinger %B upper bound for ranging


class RegimeDetector:
    """
    Detects the current market regime from the latest row of a feature DataFrame.

    Expected feature columns (produced by FeatureEngine):
        atr, atr_mean50, adx, adx_pos, adx_neg, close, ema50, bb_pct

    Usage
    -----
    >>> rd = RegimeDetector()
    >>> regime, confidence = rd.detect(df_with_features, symbol="EURUSD")
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        df_latest: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Determine the current market regime.

        Parameters
        ----------
        df_latest : pd.DataFrame
            Feature-enriched DataFrame (as returned by FeatureEngine.compute).
            Only the last row is used.
        symbol    : str, optional
            Instrument name (unused in logic, kept for future extension /
            logging context).

        Returns
        -------
        (regime_str, confidence) where confidence is in [0.0, 1.0].
        """
        if df_latest.empty:
            logger.warning("RegimeDetector.detect called with empty DataFrame")
            return NO_TRADE, 0.0

        row = df_latest.iloc[-1]

        # ── 1. Time filter: no-trade hours ────────────────────────────────────
        current_hour_utc = datetime.now(tz=timezone.utc).hour
        if current_hour_utc in settings.NO_TRADE_HOURS_UTC:
            logger.debug(
                "RegimeDetector: hour=%d is in NO_TRADE_HOURS_UTC → NO_TRADE",
                current_hour_utc,
            )
            return NO_TRADE, 1.0

        # ── 2. Extract indicator values (with NaN safety) ─────────────────────
        atr       = self._safe(row, "atr")
        atr_mean  = self._safe(row, "atr_mean50")
        adx       = self._safe(row, "adx")
        adx_pos   = self._safe(row, "adx_pos")
        adx_neg   = self._safe(row, "adx_neg")
        close     = self._safe(row, "close")
        ema50     = self._safe(row, "ema50")
        bb_pct    = self._safe(row, "bb_pct")

        # If critical values are missing, fall back safely
        if any(v is None for v in (atr, atr_mean, adx, adx_pos, adx_neg, close, ema50)):
            logger.warning(
                "RegimeDetector: one or more critical feature values are NaN "
                "→ defaulting to RANGING with low confidence"
            )
            return RANGING, 0.30

        # ── 3. VOLATILE: extreme ATR spike ────────────────────────────────────
        if atr_mean > 0 and atr > _ATR_VOLATILE_K * atr_mean:
            confidence = min(1.0, atr / (atr_mean * _ATR_VOLATILE_K + 1e-9) - 1.0)
            confidence = round(max(0.50, min(1.0, confidence)), 4)
            logger.debug(
                "RegimeDetector: VOLATILE (atr=%.5f, atr_mean50=%.5f, "
                "ratio=%.2f)",
                atr, atr_mean, atr / (atr_mean + 1e-9),
            )
            return VOLATILE, confidence

        # ── 4. TRENDING UP ────────────────────────────────────────────────────
        if (
            adx > _ADX_TREND_MIN
            and close > ema50
            and adx_pos > adx_neg
        ):
            confidence = self._trend_confidence(adx, adx_pos, adx_neg)
            logger.debug(
                "RegimeDetector: TRENDING_UP (adx=%.2f, close=%.5f, "
                "ema50=%.5f, DMP=%.2f, DMN=%.2f)",
                adx, close, ema50, adx_pos, adx_neg,
            )
            return TRENDING_UP, confidence

        # ── 5. TRENDING DOWN ─────────────────────────────────────────────────
        if (
            adx > _ADX_TREND_MIN
            and close < ema50
            and adx_neg > adx_pos
        ):
            confidence = self._trend_confidence(adx, adx_neg, adx_pos)
            logger.debug(
                "RegimeDetector: TRENDING_DOWN (adx=%.2f, close=%.5f, "
                "ema50=%.5f, DMP=%.2f, DMN=%.2f)",
                adx, close, ema50, adx_pos, adx_neg,
            )
            return TRENDING_DOWN, confidence

        # ── 6. RANGING ────────────────────────────────────────────────────────
        bb_in_range = (
            bb_pct is not None
            and _BB_PCT_LOW <= bb_pct <= _BB_PCT_HIGH
        )
        if adx < _ADX_RANGE_MAX:
            # Higher confidence when Bollinger %B also confirms consolidation
            base_conf = 0.60
            conf = base_conf + (0.20 if bb_in_range else 0.0)
            conf = round(min(1.0, conf), 4)
            logger.debug(
                "RegimeDetector: RANGING (adx=%.2f, bb_pct=%s, bb_in_range=%s)",
                adx, f"{bb_pct:.4f}" if bb_pct is not None else "N/A", bb_in_range,
            )
            return RANGING, conf

        # ── 7. Default fallback ───────────────────────────────────────────────
        logger.debug(
            "RegimeDetector: default fallback → RANGING (low confidence). "
            "adx=%.2f, close=%.5f, ema50=%.5f",
            adx, close, ema50,
        )
        return RANGING, 0.35

    @staticmethod
    def is_tradeable(regime: str) -> bool:
        """
        Returns True when opening new trades is allowed for *regime*.

        Tradeable regimes: TRENDING_UP, TRENDING_DOWN, RANGING.
        Non-tradeable: VOLATILE, NO_TRADE.
        """
        return regime in _TRADEABLE

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe(row: pd.Series, col: str):
        """Returns the scalar value for *col* or None if absent / NaN."""
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val):
            return None
        return float(val)

    @staticmethod
    def _trend_confidence(adx: float, dominant_di: float, weak_di: float) -> float:
        """
        Computes trend confidence in [0.50, 1.00] based on:
          - ADX strength (higher ADX → stronger trend)
          - DI spread (larger gap → more directional clarity)
        """
        adx_score = min(1.0, (adx - _ADX_TREND_MIN) / 25.0)   # 0 at ADX=25, 1 at ADX=50
        di_spread = dominant_di - weak_di
        di_score  = min(1.0, di_spread / 20.0)                  # 0 at 0 spread, 1 at 20

        raw = 0.50 + 0.30 * adx_score + 0.20 * di_score
        return round(min(1.0, raw), 4)
