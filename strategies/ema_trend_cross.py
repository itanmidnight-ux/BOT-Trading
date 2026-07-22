"""
Estrategia 1: Cruce de EMAs con filtro de tendencia.

Logica: EMA_FAST cruza sobre/bajo EMA_SLOW en direccion del EMA_TREND (filtro de
sesgo). Es la estrategia base de scalping en 1m: rapida a reaccionar, filtrada
para no operar contra la tendencia dominante y reducir señales falsas en rango.

Entrada BUY: EMA_FAST cruza arriba de EMA_SLOW Y close > EMA_TREND.
Entrada SELL: EMA_FAST cruza abajo de EMA_SLOW Y close < EMA_TREND.
Confianza: proporcional a la separacion entre EMA_FAST/EMA_SLOW normalizada por ATR
(cruces "fuertes" valen mas que cruces al ras).
"""
import pandas as pd

import config
from indicators import ema, atr
from strategies.base import Direction, Signal, flat_signal

NAME = "ema_trend_cross"


def generate_signal(df: pd.DataFrame) -> Signal:
    if len(df) < config.EMA_TREND + 5:
        return flat_signal(NAME, "datos insuficientes")

    close = df["close"]
    ema_fast = ema(close, config.EMA_FAST)
    ema_slow = ema(close, config.EMA_SLOW)
    ema_trend = ema(close, config.EMA_TREND)
    atr_series = atr(df, config.ATR_PERIOD)

    prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
    curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
    last_close = close.iloc[-1]
    last_trend = ema_trend.iloc[-1]
    last_atr = atr_series.iloc[-1]

    crossed_up = prev_diff <= 0 and curr_diff > 0
    crossed_down = prev_diff >= 0 and curr_diff < 0

    if last_atr <= 0 or pd.isna(last_atr):
        return flat_signal(NAME, "ATR invalido")

    strength = min(abs(curr_diff) / last_atr, 1.0)  # normaliza separacion del cruce

    if crossed_up and last_close > last_trend:
        confidence = 0.5 + 0.5 * strength
        return Signal(NAME, Direction.BUY, confidence, {
            "ema_fast": last_close, "cross": "up", "strength": strength,
        })

    if crossed_down and last_close < last_trend:
        confidence = 0.5 + 0.5 * strength
        return Signal(NAME, Direction.SELL, confidence, {
            "ema_fast": last_close, "cross": "down", "strength": strength,
        })

    return flat_signal(NAME, "sin cruce valido")
