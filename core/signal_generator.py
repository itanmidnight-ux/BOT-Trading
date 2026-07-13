from dataclasses import dataclass
from typing import Optional
import pandas as pd

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("signal_generator")


@dataclass
class Signal:
    symbol:         str
    direction:      str
    probability:    float
    atr:            float
    ensemble_votes: int = 0
    confidence:     bool = False


class SignalGenerator:
    """
    Genera una señal direccional (BUY o SELL) en cada llamada — sin HOLD
    por threshold/régimen/MTF. El único filtro que sobrevive (spread) se
    aplica en live_trader/backtester antes de invocar generate().
    """

    def __init__(self, model_updater, regime_detector, mtf_analyzer, state_manager,
                 ensemble_model=None, lstm_model=None, rl_overlay=None, ollama_advisor=None):
        self._updater  = model_updater
        self._regime   = regime_detector
        self._mtf      = mtf_analyzer
        self._state    = state_manager
        self._ensemble = ensemble_model
        self._lstm     = lstm_model
        self._rl       = rl_overlay
        self._ollama   = ollama_advisor

    def generate(self, symbol: str, df_features: pd.DataFrame,
                 feature_cols: list) -> Optional[Signal]:
        if df_features is None or len(df_features) < 2:
            return None
        if self._state.has_open_position(symbol):
            return None

        atr = float(df_features['atr'].iloc[-1]) if 'atr' in df_features.columns else 0.0
        if atr <= 0:
            return None

        latest = df_features[feature_cols].iloc[-1:].copy().fillna(0)

        # ── Ensemble (principal) ──────────────────────────────────────────────
        proba_buy      = 0.5
        confidence     = False
        ensemble_votes = 0

        if self._ensemble is not None:
            try:
                probas, conf_mask = self._ensemble.predict_with_confidence(latest)
                proba_buy  = float(probas[0])
                confidence = bool(conf_mask[0])
                weights    = self._ensemble.get_weights()
                ensemble_votes = len([w for w in weights.values() if w > 0.25])
            except Exception as e:
                _log.debug(f"Ensemble error: {e}")
                model = self._updater.get_model(symbol)
                if model is not None:
                    try:
                        proba_buy = float(model.predict_proba(latest)[0, 1])
                    except Exception:
                        return None
        else:
            model = self._updater.get_model(symbol)
            if model is None:
                return None
            try:
                proba_buy = float(model.predict_proba(latest)[0, 1])
            except Exception:
                return None

        # ── LSTM voto adicional ───────────────────────────────────────────────
        lstm_proba = 0.5
        if self._lstm is not None:
            try:
                lstm_proba = self._lstm.predict_proba(df_features.tail(25))
            except Exception:
                pass

        if self._lstm is not None and lstm_proba != 0.5:
            proba_buy = 0.70 * proba_buy + 0.30 * lstm_proba

        # ── Dirección — siempre BUY o SELL, nunca HOLD ────────────────────────
        if proba_buy >= 0.5:
            direction = constants.SIGNAL_BUY
            proba     = proba_buy
        else:
            direction = constants.SIGNAL_SELL
            proba     = 1.0 - proba_buy

        _log.info(f"SEÑAL {direction} {symbol} proba:{proba:.3f} conf:{confidence}")

        return Signal(
            symbol=symbol, direction=direction,
            probability=round(proba, 4), atr=round(atr, 6),
            ensemble_votes=ensemble_votes,
            confidence=confidence,
        )
