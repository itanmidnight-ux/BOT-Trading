"""Tests de importación de todos los módulos."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock MetaTrader5 antes de importar módulos que lo usan
from unittest.mock import MagicMock, patch
sys.modules['MetaTrader5'] = MagicMock()

def test_config_constants():
    from config.constants import BOT_VERSION, REGIME_TRENDING_UP, SIGNAL_BUY
    assert BOT_VERSION == "3.0.0"
    assert REGIME_TRENDING_UP == "TRENDING_UP"
    assert SIGNAL_BUY == "BUY"

def test_config_settings():
    from config.settings import INITIAL_CAPITAL, SYMBOL_PHASE1, MIN_WIN_RATE_LIVE
    assert INITIAL_CAPITAL == 20.0
    assert SYMBOL_PHASE1 == "EURUSD"
    assert MIN_WIN_RATE_LIVE == 0.59

def test_config_settings_scalp():
    from config.settings import SYMBOL, LEVERAGE_XAUUSD, MAX_SPREAD_USD, ATR_SL_MULTIPLIER, ATR_TP1_MULTIPLIER
    assert SYMBOL == "XAUUSD"
    assert LEVERAGE_XAUUSD == 1
    assert MAX_SPREAD_USD == 0.35
    assert ATR_SL_MULTIPLIER == 1.0
    assert ATR_TP1_MULTIPLIER == 1.2

def test_utils_logger():
    from utils.logger import get_logger
    log = get_logger("test")
    assert log is not None

def test_utils_display():
    from utils.display import print_banner, print_menu
    assert callable(print_banner)
    assert callable(print_menu)

def test_utils_notifier():
    from utils.notifier import notify, TRADE_OPEN, TRADE_CLOSE
    assert TRADE_OPEN == "TRADE_OPEN"
    assert callable(notify)

def test_core_feature_engine():
    from core.feature_engine import FeatureEngine
    fe = FeatureEngine()
    cols = fe.get_feature_cols()
    assert len(cols) > 15
    assert 'rsi' in cols
    assert 'atr' in cols
    assert 'macd' in cols

def test_core_regime_detector():
    from core.regime_detector import RegimeDetector
    rd = RegimeDetector()
    assert callable(rd.detect)
    assert callable(rd.is_tradeable)

def test_core_multi_tf():
    from core.multi_tf_analyzer import MultiTFAnalyzer
    mtf = MultiTFAnalyzer()
    assert callable(mtf.get_consensus)

def test_core_signal_generator():
    from core.signal_generator import SignalGenerator, Signal
    assert callable(SignalGenerator)

def test_core_exit_manager():
    from core.exit_manager import ExitManager, ExitAction, OpenPosition
    em = ExitManager()
    assert callable(em.evaluate)
    assert callable(em.calc_levels)

def test_core_state_manager():
    from core.state_manager import StateManager
    sm = StateManager()
    assert not sm.has_open_position("EURUSD")
    assert sm.capital >= 0  # puede tener capital de sesión previa

def test_core_backtester():
    from core.backtester import Backtester
    bt = Backtester("EURUSD")
    assert callable(bt.run)

def test_core_training_loop():
    from core.training_loop import TrainingLoop
    assert callable(TrainingLoop)

def test_core_auto_improver():
    from core.auto_improver import AutoImprover
    imp = AutoImprover()
    assert callable(imp.run)

def test_analysis_performance_reporter():
    from analysis.performance_reporter import PerformanceReporter
    rp = PerformanceReporter()
    assert callable(rp.show)

def test_analysis_log_analyzer():
    from analysis.log_analyzer import LogAnalyzer
    la = LogAnalyzer()
    df = la.load_all_trades("EURUSD")
    assert df is not None

def test_analysis_optimizer():
    from analysis.optimizer import Optimizer
    opt = Optimizer()
    assert callable(opt.optimize)
