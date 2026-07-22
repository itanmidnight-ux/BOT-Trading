"""Regresiones de la pasada de revision: cada test cubre un bug corregido."""
import numpy as np
import pandas as pd

from indicators import vwap, williams_fractals
from profit_manager import ActionType, ProfitManager


def test_vwap_works_without_time_column(trending_df):
    df = trending_df.drop(columns=["time"]).set_index(trending_df["time"])
    result = vwap(df)
    assert len(result) == len(df)
    assert result.notna().all()


def test_fractals_detect_local_extremes():
    # Pico claro en la posicion 5 y valle claro en la posicion 10.
    high = [10, 10, 10, 10, 11, 15, 11, 10, 10, 10, 9, 10, 10, 10, 10]
    low = [h - 1 for h in high]
    low[10] = 4
    df = pd.DataFrame({"high": high, "low": low})

    up, down = williams_fractals(df, window=2)

    assert up.iloc[5]
    assert down.iloc[10]
    assert up.sum() == 1
    assert down.sum() == 1


def test_fractals_edges_are_never_fractals(trending_df):
    up, down = williams_fractals(trending_df, window=2)
    assert not up.iloc[:2].any() and not up.iloc[-2:].any()
    assert not down.iloc[:2].any() and not down.iloc[-2:].any()


def test_profit_manager_has_position():
    pm = ProfitManager()
    assert not pm.has_position(7)
    pm.register_position(7, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1950.0, tp2_price=1980.0)
    assert pm.has_position(7)
    pm.forget_position(7)
    assert not pm.has_position(7)


def test_trailing_respects_broker_min_stop_distance():
    pm = ProfitManager()
    pm.register_position(1, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1950.0, tp2_price=1980.0)

    min_stop = 5.0  # mayor que la distancia de trailing por ATR (1.0 * atr=2.0)
    actions = pm.evaluate(1, current_price=1910.0, current_sl=1895.0, atr_value=2.0,
                          min_stop_distance=min_stop)

    modify = [a for a in actions if a.type == ActionType.MODIFY_SL]
    assert modify, "se esperaba un trailing modify"
    assert 1910.0 - modify[0].sl_price >= min_stop


def test_backtest_charges_spread_cost(make_df):
    from backtester import run_backtest

    df = make_df(n=400, seed=7)
    cheap = run_backtest(df, initial_balance=1000.0, spread_points=0.0)
    expensive = run_backtest(df, initial_balance=1000.0, spread_points=200.0)

    # Mismos datos y señales: mas spread nunca puede dar mejor resultado.
    if cheap.total_trades > 0 and expensive.total_trades > 0:
        assert expensive.final_balance <= cheap.final_balance
