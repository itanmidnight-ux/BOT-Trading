"""
Indicadores tecnicos puros pandas/numpy (sin TA-Lib, para evitar la dependencia
binaria en despliegues Linux/Wine). Todas las funciones son deterministas y
operan sobre un DataFrame con columnas: open, high, low, close, tick_volume.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def bollinger_bands(series: pd.Series, period: int, num_std: float):
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def vwap(df: pd.DataFrame, session_reset_hour_utc: int = 0) -> pd.Series:
    """VWAP intradia, reseteado en cada cambio de sesion (hora UTC configurable).
    Requiere columna datetime `time` (UTC) en el DataFrame o index datetime."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["tick_volume"].replace(0, 1)  # evita division por cero en velas sin volumen reportado

    if "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True)
    else:
        ts = pd.to_datetime(df.index, utc=True)

    session_id = ((ts.dt.tz_localize(None) - pd.Timedelta(hours=session_reset_hour_utc)).dt.date)
    session_id = pd.Series(session_id.values, index=df.index)

    pv = typical_price * volume
    cum_pv = pv.groupby(session_id).cumsum()
    cum_vol = volume.groupby(session_id).cumsum()
    return cum_pv / cum_vol


def williams_fractals(df: pd.DataFrame, window: int = 2):
    """Fractales de Bill Williams: devuelve (fractal_up, fractal_down) booleanos.
    Un fractal alcista (resistencia) es un high mayor que `window` velas a cada lado;
    uno bajista (soporte), un low menor que `window` velas a cada lado."""
    high, low = df["high"], df["low"]
    n = len(df)
    fractal_up = pd.Series(False, index=df.index)
    fractal_down = pd.Series(False, index=df.index)

    for i in range(window, n - window):
        h_slice = high.iloc[i - window: i + window + 1]
        l_slice = low.iloc[i - window: i + window + 1]
        if high.iloc[i] == h_slice.max() and (h_slice == h_slice.max()).sum() == 1:
            fractal_up.iloc[i] = True
        if low.iloc[i] == l_slice.min() and (l_slice == l_slice.min()).sum() == 1:
            fractal_down.iloc[i] = True

    return fractal_up, fractal_down
