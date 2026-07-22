"""Fixtures compartidas: datos OHLC sinteticos y symbol_info simulado, para
poder testear la logica del bot sin conexion real a MT5."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _make_trending_df(n=200, start=1900.0, step=0.05, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    increments = np.full(n, step) + rng.normal(0, noise, n)
    close = start + np.cumsum(increments)
    open_ = np.roll(close, 1)
    open_[0] = start
    # Padding intrabar (high/low) generoso para producir un ATR de magnitud
    # realista para XAUUSD M1 (~1 unidad de precio), independiente del ruido
    # de la caminata de `close` (que se mantiene chico para no romper la
    # tendencia que exigen ema_trend_cross/vwap_momentum).
    high = np.maximum(open_, close) + rng.uniform(0.3, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.3, 1.0, n)
    volume = rng.integers(50, 200, n)
    time = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": time, "open": open_, "high": high, "low": low,
        "close": close, "tick_volume": volume,
    })


@pytest.fixture
def make_df():
    return _make_trending_df


@pytest.fixture
def trending_df():
    return _make_trending_df()


@pytest.fixture
def symbol_info():
    return SimpleNamespace(
        point=0.01, digits=2, trade_stops_level=50, trade_tick_value=1.0,
        trade_tick_size=0.01, trade_contract_size=100.0,
        volume_step=0.01, volume_min=0.01, volume_max=50.0,
    )
