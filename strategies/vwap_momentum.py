"""
Estrategia 4: VWAP + momentum (MACD).

Logica: usa el VWAP intradia como referencia de "valor justo" de la sesion.
Solo opera a favor de donde esta el precio respecto al VWAP, y exige que el
histograma de MACD confirme momentum en esa misma direccion (cruzando cero o
expandiendose). Es la estrategia mas sensible al flujo institucional/order flow
aproximado por volumen, complementa a las 3 anteriores que son puramente de precio.

Entrada BUY: close > VWAP Y histograma MACD cruza de negativo a positivo (o ya positivo y creciendo).
Entrada SELL: close < VWAP Y histograma MACD cruza de positivo a negativo (o ya negativo y decreciendo).
Confianza: combina distancia al VWAP (normalizada por ATR) y pendiente del histograma.
"""
import pandas as pd

import config
from indicators import atr, macd, vwap
from strategies.base import Direction, Signal, flat_signal

NAME = "vwap_momentum"


def generate_signal(df: pd.DataFrame) -> Signal:
    if len(df) < config.MACD_SLOW + config.MACD_SIGNAL + 5:
        return flat_signal(NAME, "datos insuficientes")

    close = df["close"]
    atr_series = atr(df, config.ATR_PERIOD)
    vwap_series = vwap(df, config.VWAP_SESSION_RESET_HOUR_UTC)
    _, _, hist = macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)

    last_close = close.iloc[-1]
    last_vwap = vwap_series.iloc[-1]
    last_atr = atr_series.iloc[-1]
    curr_hist, prev_hist = hist.iloc[-1], hist.iloc[-2]

    if pd.isna(last_vwap) or last_atr <= 0 or pd.isna(last_atr):
        return flat_signal(NAME, "VWAP/ATR invalido")

    vwap_distance = (last_close - last_vwap) / last_atr
    momentum_rising = curr_hist > prev_hist

    if last_close > last_vwap and curr_hist > 0 and momentum_rising:
        confidence = 0.5 + 0.25 * min(abs(vwap_distance), 1.0) + 0.25 * min(abs(curr_hist - prev_hist) / last_atr, 1.0)
        return Signal(NAME, Direction.BUY, min(confidence, 1.0), {
            "vwap": last_vwap, "hist": curr_hist,
        })

    if last_close < last_vwap and curr_hist < 0 and not momentum_rising:
        confidence = 0.5 + 0.25 * min(abs(vwap_distance), 1.0) + 0.25 * min(abs(curr_hist - prev_hist) / last_atr, 1.0)
        return Signal(NAME, Direction.SELL, min(confidence, 1.0), {
            "vwap": last_vwap, "hist": curr_hist,
        })

    return flat_signal(NAME, "sin confirmacion de momentum")
