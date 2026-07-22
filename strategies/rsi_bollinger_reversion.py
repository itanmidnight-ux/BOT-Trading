"""
Estrategia 2: Reversion a la media con RSI + Bandas de Bollinger.

Logica: opera reversiones cuando el precio toca/perfora una banda de Bollinger
EN SIMULTANEO con RSI en zona de sobrecompra/sobreventa, y la vela mas reciente
ya muestra rechazo (cierre volviendo hacia la media). Es la contraparte de
`ema_trend_cross`: en vez de seguir tendencia, captura extremos de corto plazo,
muy frecuentes en el ruido de XAUUSD 1m.

Entrada BUY: low <= banda_inferior, RSI <= RSI_OVERSOLD, vela actual cierra > apertura.
Entrada SELL: high >= banda_superior, RSI >= RSI_OVERBOUGHT, vela actual cierra < apertura.
Confianza: combina que tan lejos esta el RSI del umbral y que tan clara es la vela de rechazo.
"""
import pandas as pd

import config
from indicators import rsi, bollinger_bands
from strategies.base import Direction, Signal, flat_signal

NAME = "rsi_bollinger_reversion"


def generate_signal(df: pd.DataFrame) -> Signal:
    if len(df) < config.BOLLINGER_PERIOD + 5:
        return flat_signal(NAME, "datos insuficientes")

    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    rsi_series = rsi(close, config.RSI_PERIOD)
    upper, mid, lower = bollinger_bands(close, config.BOLLINGER_PERIOD, config.BOLLINGER_STD)

    last_rsi = rsi_series.iloc[-1]
    last_close, last_open = close.iloc[-1], open_.iloc[-1]
    last_low, last_high = low.iloc[-1], high.iloc[-1]
    last_lower, last_upper = lower.iloc[-1], upper.iloc[-1]

    if pd.isna(last_lower) or pd.isna(last_upper):
        return flat_signal(NAME, "bandas invalidas")

    bullish_rejection = last_close > last_open
    bearish_rejection = last_close < last_open

    if last_low <= last_lower and last_rsi <= config.RSI_OVERSOLD and bullish_rejection:
        rsi_strength = min((config.RSI_OVERSOLD - last_rsi) / config.RSI_OVERSOLD, 1.0) if config.RSI_OVERSOLD > 0 else 0
        confidence = 0.5 + 0.5 * max(rsi_strength, 0.0)
        return Signal(NAME, Direction.BUY, confidence, {
            "rsi": last_rsi, "band": "lower", "close": last_close,
        })

    if last_high >= last_upper and last_rsi >= config.RSI_OVERBOUGHT and bearish_rejection:
        rsi_strength = min((last_rsi - config.RSI_OVERBOUGHT) / (100 - config.RSI_OVERBOUGHT), 1.0) if config.RSI_OVERBOUGHT < 100 else 0
        confidence = 0.5 + 0.5 * max(rsi_strength, 0.0)
        return Signal(NAME, Direction.SELL, confidence, {
            "rsi": last_rsi, "band": "upper", "close": last_close,
        })

    return flat_signal(NAME, "sin extremo con rechazo")
