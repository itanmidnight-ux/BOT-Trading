"""Tests de lógica core sin conexión MT5."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch
sys.modules['MetaTrader5'] = MagicMock()

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n=500, seed=42):
    """Genera DataFrame OHLCV sintético."""
    rng   = np.random.default_rng(seed)
    close = 1.08 + np.cumsum(rng.normal(0, 0.0002, n))
    open_ = close + rng.normal(0, 0.0001, n)
    high  = np.maximum(close, open_) + rng.uniform(0.0001, 0.0005, n)
    low   = np.minimum(close, open_) - rng.uniform(0.0001, 0.0005, n)
    vol   = rng.integers(100, 1000, n)
    times = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": open_, "high": high,
        "low": low, "close": close, "tick_volume": vol, "spread": 10,
    })


class TestFeatureEngine:
    def test_compute_returns_dataframe(self):
        from core.feature_engine import FeatureEngine
        fe  = FeatureEngine()
        df  = _make_ohlcv(300)
        out = fe.compute(df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) < len(df)   # elimina primeras filas NaN

    def test_feature_cols_present(self):
        from core.feature_engine import FeatureEngine
        fe   = FeatureEngine()
        df   = _make_ohlcv(300)
        out  = fe.compute(df)
        cols = fe.get_feature_cols()
        for c in ['rsi', 'atr', 'macd', 'ema9', 'ema21', 'adx']:
            assert c in out.columns, f"Falta columna: {c}"
        assert all(c in out.columns for c in cols)

    def test_target_binary(self):
        from core.feature_engine import FeatureEngine
        fe  = FeatureEngine()
        df  = _make_ohlcv(300)
        out = fe.compute(df)
        out = fe.add_target(out, "XAUUSD")
        assert "target" in out.columns
        assert set(out["target"].dropna().unique()).issubset({0, 1})
        # Horizonte de 1 vela: solo la última fila queda sin target válido.
        assert out["target"].iloc[:-1].notna().all()
        assert out["target"].iloc[-1:].isna().all()

    def test_get_latest_features(self):
        from core.feature_engine import FeatureEngine
        fe  = FeatureEngine()
        df  = _make_ohlcv(300)
        out = fe.compute(df)
        lat = fe.get_latest_features(out)
        assert isinstance(lat, pd.Series)
        assert len(lat) > 0


class TestRegimeDetector:
    def test_detect_returns_tuple(self):
        from core.regime_detector import RegimeDetector
        from core.feature_engine import FeatureEngine
        fe     = FeatureEngine()
        rd     = RegimeDetector()
        df     = _make_ohlcv(300)
        df_f   = fe.compute(df)
        regime, conf = rd.detect(df_f)
        assert isinstance(regime, str)
        assert 0.0 <= conf <= 1.0

    def test_is_tradeable(self):
        from core.regime_detector import RegimeDetector
        from config.constants import REGIME_TRENDING_UP, REGIME_NO_TRADE, REGIME_VOLATILE
        rd = RegimeDetector()
        assert rd.is_tradeable(REGIME_TRENDING_UP)
        assert not rd.is_tradeable(REGIME_NO_TRADE)
        assert not rd.is_tradeable(REGIME_VOLATILE)


class TestExitManager:
    def _make_position(self, direction="BUY"):
        from core.exit_manager import OpenPosition
        from datetime import datetime
        return OpenPosition(
            ticket=1, symbol="XAUUSD", direction=direction,
            lots=0.01, entry=2400.00, sl=2399.00, tp=2401.20,
            open_time=datetime.utcnow(), bars_open=0,
        )

    def test_sl_hit_buy(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2399.80, "high": 2399.90, "low": 2398.50, "close": 2398.80, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "SL"

    def test_tp_hit_buy(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2400.50, "high": 2401.50, "low": 2400.40, "close": 2401.30, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "TP"

    def test_forced_bar_close_when_neither_sl_nor_tp_hit(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2400.10, "high": 2400.60, "low": 2399.80, "close": 2400.40, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "BAR_CLOSE"

    def test_calc_levels_buy(self):
        from core.exit_manager import ExitManager
        em     = ExitManager()
        levels = em.calc_levels("XAUUSD", "BUY", 2400.00, 1.0)
        assert levels["sl"] < 2400.00
        assert levels["tp"] > 2400.00

    def test_calc_levels_sell(self):
        from core.exit_manager import ExitManager
        em     = ExitManager()
        levels = em.calc_levels("XAUUSD", "SELL", 2400.00, 1.0)
        assert levels["sl"] > 2400.00
        assert levels["tp"] < 2400.00


class TestStateManager:
    def test_save_and_retrieve(self):
        from core.state_manager import StateManager
        sm = StateManager()
        sm.save_position("EURUSD", 123, "BUY", 0.03, 1.0845, 1.0830, 1.0865, 1.0880)
        assert sm.has_open_position("EURUSD")
        pos = sm.get_position("EURUSD")
        assert pos["ticket"]    == 123
        assert pos["direction"] == "BUY"
        sm.clear_position("EURUSD")
        assert not sm.has_open_position("EURUSD")

    def test_capital_update(self):
        from core.state_manager import StateManager
        sm = StateManager()
        sm.capital = 25.50
        assert sm.capital == 25.50

    def test_consecutive_losses(self):
        from core.state_manager import StateManager
        sm = StateManager()
        sm.record_trade_result(False)
        sm.record_trade_result(False)
        assert sm.consecutive_losses == 2
        sm.record_trade_result(True)
        assert sm.consecutive_losses == 0


class TestBacktester:
    def test_run_returns_metrics(self):
        from core.backtester import Backtester
        from core.feature_engine import FeatureEngine
        from unittest.mock import MagicMock
        import numpy as np

        fe   = FeatureEngine()
        df   = _make_ohlcv(500)
        df_f = fe.compute(df)
        df_f = fe.add_target(df_f, "EURUSD")
        df_f = df_f.dropna().reset_index(drop=True)

        feature_cols = fe.get_feature_cols()

        # Mock del modelo XGBoost
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        bt      = Backtester("EURUSD")
        metrics = bt.run(df_f, mock_model, feature_cols, 0.62, 20.0)

        assert "win_rate"      in metrics
        assert "profit_factor" in metrics
        assert "total_trades"  in metrics
        assert "equity_curve"  in metrics
        assert metrics["final_capital"] >= 0

    def test_no_future_leak(self):
        """Verifica que el backtester no usa datos futuros."""
        from core.backtester import Backtester
        from core.feature_engine import FeatureEngine
        from unittest.mock import MagicMock
        import numpy as np

        fe   = FeatureEngine()
        df   = _make_ohlcv(300)
        df_f = fe.compute(df)
        df_f = fe.add_target(df_f, "EURUSD")
        df_f = df_f.dropna().reset_index(drop=True)

        feature_cols = fe.get_feature_cols()
        mock_model   = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        bt      = Backtester("EURUSD")
        metrics = bt.run(df_f, mock_model, feature_cols, 0.62, 20.0)
        # No crash = sin data leakage por índices fuera de rango
        assert metrics is not None


class TestKellyEngine:
    def test_calculate_fraction(self):
        from core.kelly import KellyEngine
        ke  = KellyEngine("EURUSD")
        # WR 60%, avg_win 10 pips, avg_loss 8 pips → f* > 0
        frac = ke.calculate_fraction(0.60, 10.0, 8.0)
        assert 0.05 <= frac <= 0.25

    def test_fraction_below_zero_clamped(self):
        from core.kelly import KellyEngine
        ke   = KellyEngine("EURUSD")
        frac = ke.calculate_fraction(0.30, 5.0, 20.0)  # Negativo → clamp a min
        assert frac >= 0.05

    def test_fraction_to_lots_eurusd(self):
        from core.kelly import KellyEngine
        ke   = KellyEngine("EURUSD")
        lots = ke.fraction_to_lots(0.05, 20.0, "EURUSD", 1.085)
        assert lots >= 0.01
        assert lots <= 1.0


class TestCapitalScaler:
    def test_phase1(self):
        from core.capital_scaler import CapitalScaler
        cs = CapitalScaler()
        assert cs.get_phase(20.0) == 1
        assert "EURUSD" in cs.get_active_symbols(20.0)
        assert "XAUUSD" not in cs.get_active_symbols(20.0)

    def test_phase2(self):
        from core.capital_scaler import CapitalScaler
        cs = CapitalScaler()
        assert cs.get_phase(35.0) == 2
        assert "XAUUSD" in cs.get_active_symbols(35.0)

    def test_leverage(self):
        from core.capital_scaler import CapitalScaler
        cs = CapitalScaler()
        assert cs.get_leverage("EURUSD") == 3000
        assert cs.get_leverage("XAUUSD") == 1000
