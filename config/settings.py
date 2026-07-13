import json
import os
from pathlib import Path

_BASE = Path(__file__).parent.parent

def _load_runtime() -> dict:
    p = _BASE / "config" / "runtime_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def _load_optimized() -> dict:
    p = _BASE / "config" / "optimized_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

_rt  = _load_runtime()
_opt = _load_optimized()
_merged = {**_opt, **_rt}

def _get(key, default):
    return _merged.get(key, default)

# ── Instruments ────────────────────────────────────────────────────────────────
SYMBOL_PHASE1              = _get("SYMBOL_PHASE1", "EURUSD")
SYMBOL_PHASE2              = _get("SYMBOL_PHASE2", "XAUUSD")
CAPITAL_PHASE2_THRESHOLD   = _get("CAPITAL_PHASE2_THRESHOLD", 30.0)
LEVERAGE_PHASE1            = _get("LEVERAGE_PHASE1", 3000)
LEVERAGE_PHASE2            = _get("LEVERAGE_PHASE2", 1000)
SYMBOL                     = _get("SYMBOL", "XAUUSD")
LEVERAGE_XAUUSD            = _get("LEVERAGE_XAUUSD", 1)

# ── Timeframes ─────────────────────────────────────────────────────────────────
TIMEFRAME_MAIN     = "M1"
TIMEFRAMES_CONFIRM = ["M5", "M15"]
ALL_TIMEFRAMES     = ["M1", "M5", "M15"]

# ── Capital & Kelly ────────────────────────────────────────────────────────────
INITIAL_CAPITAL        = _get("INITIAL_CAPITAL", 20.0)
KELLY_MAX_FRACTION     = _get("KELLY_MAX_FRACTION", 0.20)
KELLY_MIN_FRACTION     = _get("KELLY_MIN_FRACTION", 0.05)
KELLY_MIN_LOTS         = _get("KELLY_MIN_LOTS", 0.01)
KELLY_MAX_LOTS         = _get("KELLY_MAX_LOTS", 1.0)
KELLY_RECALC_EVERY     = _get("KELLY_RECALC_EVERY", 20)
KELLY_BOOTSTRAP_FRAC   = _get("KELLY_BOOTSTRAP_FRAC", 0.05)

# ── ATR Multipliers ────────────────────────────────────────────────────────────
ATR_SL_MULTIPLIER       = _get("ATR_SL_MULTIPLIER", 1.0)
ATR_TP1_MULTIPLIER      = _get("ATR_TP1_MULTIPLIER", 1.2)

# ── Risk Management ────────────────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT       = _get("MAX_DAILY_LOSS_PCT", 0.05)
MAX_DRAWDOWN_PCT         = _get("MAX_DRAWDOWN_PCT", 0.15)
MAX_CONSECUTIVE_LOSSES   = _get("MAX_CONSECUTIVE_LOSSES", 5)
MIN_MARGIN_BUFFER        = _get("MIN_MARGIN_BUFFER", 2.0)
MAX_SPREAD_USD           = _get("MAX_SPREAD_USD", 0.35)

# ── ML Model ───────────────────────────────────────────────────────────────────
SIGNAL_THRESHOLD          = _get("SIGNAL_THRESHOLD", 0.62)
RETRAIN_EVERY_N_CANDLES   = _get("RETRAIN_EVERY_N_CANDLES", 500)
MIN_CANDLES_TO_TRAIN      = _get("MIN_CANDLES_TO_TRAIN", 100000)
WALK_FORWARD_TRAIN        = _get("WALK_FORWARD_TRAIN", 0.70)
WALK_FORWARD_VAL          = _get("WALK_FORWARD_VAL", 0.15)
WALK_FORWARD_TEST         = _get("WALK_FORWARD_TEST", 0.15)

XGB_PARAMS = {
    "n_estimators":         _get("xgb_n_estimators", 600),
    "max_depth":            _get("xgb_max_depth", 5),
    "learning_rate":        _get("xgb_learning_rate", 0.015),
    "subsample":            _get("xgb_subsample", 0.8),
    "colsample_bytree":     _get("xgb_colsample_bytree", 0.8),
    "min_child_weight":     _get("xgb_min_child_weight", 3),
    "gamma":                _get("xgb_gamma", 0.1),
    "eval_metric":          "logloss",
    "early_stopping_rounds": 50,
    "use_label_encoder":    False,
    "verbosity":            0,
}

# ── Spread simulation (backtest) ───────────────────────────────────────────────
SPREAD_POINTS_EURUSD = _get("SPREAD_POINTS_EURUSD", 10)
SPREAD_POINTS_XAUUSD = _get("SPREAD_POINTS_XAUUSD", 30)

# ── Time filters ───────────────────────────────────────────────────────────────
NO_TRADE_HOURS_UTC = _get("NO_TRADE_HOURS_UTC", [22, 23, 0, 1])

# ── Training loop gate ─────────────────────────────────────────────────────────
MIN_WIN_RATE_LIVE = _get("MIN_WIN_RATE_LIVE", 0.59)
TARGET_WIN_RATE   = _get("TARGET_WIN_RATE", 0.70)
MAX_TRAIN_ITERS   = _get("MAX_TRAIN_ITERS", 20)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = _BASE / "data" / "raw"
DATA_PROC_DIR      = _BASE / "data" / "processed"
MODELS_DIR         = _BASE / "models" / "checkpoints"
REPORTS_DIR        = _BASE / "models" / "reports"
LOGS_TRADES_DIR    = _BASE / "logs" / "trades"
LOGS_BACKTEST_DIR  = _BASE / "logs" / "backtest"
LOGS_SYSTEM_DIR    = _BASE / "logs" / "system"
CONFIG_DIR         = _BASE / "config"
