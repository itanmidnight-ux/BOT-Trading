"""
Estrategia 3: Ruptura de fractales de Williams (accion del precio).

Logica: identifica el ultimo fractal alcista (resistencia local) y el ultimo
fractal bajista (soporte local) confirmados. Si el precio actual rompe ese nivel
por al menos `BREAKOUT_MIN_ATR_MULTIPLE * ATR` (para filtrar rupturas de ruido),
se considera breakout valido.

Entrada BUY: close > ultimo fractal alcista + colchon ATR.
Entrada SELL: close < ultimo fractal bajista - colchon ATR.
Confianza: proporcional a cuanto excede el precio el nivel roto, en unidades de ATR.
"""
import pandas as pd

import config
from indicators import atr, williams_fractals
from strategies.base import Direction, Signal, flat_signal

NAME = "fractal_breakout"


def generate_signal(df: pd.DataFrame) -> Signal:
    min_len = config.FRACTAL_WINDOW * 2 + config.ATR_PERIOD + 5
    if len(df) < min_len:
        return flat_signal(NAME, "datos insuficientes")

    close = df["close"]
    atr_series = atr(df, config.ATR_PERIOD)
    fractal_up, fractal_down = williams_fractals(df, config.FRACTAL_WINDOW)

    # Excluye las ultimas `window` velas: un fractal necesita velas futuras para confirmarse.
    cutoff = len(df) - config.FRACTAL_WINDOW
    confirmable_df = df.iloc[:cutoff]
    up_levels = confirmable_df[fractal_up.iloc[:cutoff].to_numpy()]
    down_levels = confirmable_df[fractal_down.iloc[:cutoff].to_numpy()]

    last_close = close.iloc[-1]
    last_atr = atr_series.iloc[-1]
    if last_atr <= 0 or pd.isna(last_atr):
        return flat_signal(NAME, "ATR invalido")

    buffer = config.BREAKOUT_MIN_ATR_MULTIPLE * last_atr

    if not up_levels.empty:
        last_resistance = up_levels["high"].iloc[-1]
        if last_close > last_resistance + buffer:
            excess = (last_close - last_resistance) / last_atr
            confidence = min(0.5 + 0.25 * excess, 1.0)
            return Signal(NAME, Direction.BUY, confidence, {
                "resistance": last_resistance, "close": last_close,
            })

    if not down_levels.empty:
        last_support = down_levels["low"].iloc[-1]
        if last_close < last_support - buffer:
            excess = (last_support - last_close) / last_atr
            confidence = min(0.5 + 0.25 * excess, 1.0)
            return Signal(NAME, Direction.SELL, confidence, {
                "support": last_support, "close": last_close,
            })

    return flat_signal(NAME, "sin ruptura de fractal")
