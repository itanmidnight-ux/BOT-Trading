"""
Parametros configurables del bot XAUUSD 1m.
Ningun valor de estrategia esta hardcodeado en la logica: todo se lee desde aqui
(o se sobreescribe via variables de entorno) para poder ajustar sin tocar codigo.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Instrumento / timeframe
# ---------------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M1")          # ventana operativa
CONTEXT_TIMEFRAME = os.getenv("CONTEXT_TIMEFRAME", "M15")  # filtro de tendencia superior
BARS_LOOKBACK = _env_int("BARS_LOOKBACK", 500)    # velas M1 cargadas por ciclo

# ---------------------------------------------------------------------------
# Credenciales / conexion MT5 (dejar vacio -> usa la sesion ya logueada en el terminal)
# ---------------------------------------------------------------------------
MT5_LOGIN = os.getenv("MT5_LOGIN")
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")  # ruta al terminal64.exe, opcional
MT5_RECONNECT_ATTEMPTS = _env_int("MT5_RECONNECT_ATTEMPTS", 5)
MT5_RECONNECT_BACKOFF_SEC = _env_float("MT5_RECONNECT_BACKOFF_SEC", 3.0)

# ---------------------------------------------------------------------------
# Bridge EA <-> Python (comunicacion por archivos, ver mt5_connector/bridge.py)
# ---------------------------------------------------------------------------
USE_EA_BRIDGE = _env_bool("USE_EA_BRIDGE", False)  # False = ordenes via API python directa
MT5_COMMON_FILES_PATH = os.getenv("MT5_COMMON_FILES_PATH", "")  # .../Terminal/Common/Files
BRIDGE_SUBDIR = "bot_bridge"
BRIDGE_MAGIC = _env_int("BRIDGE_MAGIC", 990099)
BRIDGE_POLL_TIMEOUT_SEC = _env_float("BRIDGE_POLL_TIMEOUT_SEC", 5.0)
BRIDGE_POLL_INTERVAL_SEC = _env_float("BRIDGE_POLL_INTERVAL_SEC", 0.2)

# ---------------------------------------------------------------------------
# Capital / position sizing dinamico
# ---------------------------------------------------------------------------
RISK_PER_TRADE_PCT = _env_float("RISK_PER_TRADE_PCT", 0.5)      # % del equity arriesgado por trade
MAX_RISK_PER_TRADE_PCT = _env_float("MAX_RISK_PER_TRADE_PCT", 1.0)  # techo duro, aunque config suba
MIN_LOT = _env_float("MIN_LOT", 0.01)
MAX_CAPITAL_ALLOCATION_PCT = _env_float("MAX_CAPITAL_ALLOCATION_PCT", 50.0)  # % equity en margen usado simultaneo

# ---------------------------------------------------------------------------
# Riesgo global
# ---------------------------------------------------------------------------
MIN_RR_RATIO = _env_float("MIN_RR_RATIO", 2.0)
MAX_OPEN_POSITIONS = _env_int("MAX_OPEN_POSITIONS", 5)
MAX_DAILY_LOSS_PCT = _env_float("MAX_DAILY_LOSS_PCT", 5.0)
MAX_DRAWDOWN_PCT = _env_float("MAX_DRAWDOWN_PCT", 15.0)

# ---------------------------------------------------------------------------
# Stop loss dinamico (ATR)
# ---------------------------------------------------------------------------
ATR_PERIOD = _env_int("ATR_PERIOD", 14)
ATR_SL_MULTIPLIER = _env_float("ATR_SL_MULTIPLIER", 1.5)
MIN_SL_BUFFER_POINTS = _env_float("MIN_SL_BUFFER_POINTS", 50)   # colchon sobre stops_level del broker
MAX_SPREAD_POINTS = _env_float("MAX_SPREAD_POINTS", 350)        # no operar si el spread excede esto
SLIPPAGE_POINTS = _env_int("SLIPPAGE_POINTS", 30)

# ---------------------------------------------------------------------------
# Gestion de ganancias / trailing / TP escalonado
# ---------------------------------------------------------------------------
ATR_TP1_MULTIPLIER = _env_float("ATR_TP1_MULTIPLIER", 1.5)   # primer parcial
ATR_TP2_MULTIPLIER = _env_float("ATR_TP2_MULTIPLIER", 3.0)   # segundo parcial
TP1_CLOSE_FRACTION = _env_float("TP1_CLOSE_FRACTION", 0.5)   # % de volumen cerrado en TP1
TP2_CLOSE_FRACTION = _env_float("TP2_CLOSE_FRACTION", 0.3)   # % adicional cerrado en TP2 (resto = trailing libre)

MIN_PROFIT_TO_ARM_TRAILING_USD = _env_float("MIN_PROFIT_TO_ARM_TRAILING_USD", 0.0)  # 0 = usa ATR_TRAIL_ARM_MULTIPLIER
ATR_TRAIL_ARM_MULTIPLIER = _env_float("ATR_TRAIL_ARM_MULTIPLIER", 0.8)  # profit minimo (en ATR) antes de armar trailing
ATR_TRAIL_DISTANCE_MULTIPLIER = _env_float("ATR_TRAIL_DISTANCE_MULTIPLIER", 1.0)  # distancia del trailing al peak
PROFIT_GIVEBACK_TOLERANCE_PCT = _env_float("PROFIT_GIVEBACK_TOLERANCE_PCT", 25.0)  # % del peak profit que se tolera perder

# ---------------------------------------------------------------------------
# Grid de contencion
# ---------------------------------------------------------------------------
GRID_ENABLED = _env_bool("GRID_ENABLED", True)
GRID_LEVELS = _env_int("GRID_LEVELS", 4)
GRID_STEP_ATR_MULTIPLIER = _env_float("GRID_STEP_ATR_MULTIPLIER", 0.75)
GRID_LOT_MULTIPLIER = _env_float("GRID_LOT_MULTIPLIER", 1.3)   # escalado geometrico moderado por nivel
GRID_MAX_TOTAL_RISK_PCT = _env_float("GRID_MAX_TOTAL_RISK_PCT", 3.0)  # riesgo agregado maximo del grid completo

# ---------------------------------------------------------------------------
# Estrategias / confluencia
# ---------------------------------------------------------------------------
STRATEGIES_ENABLED = {
    "ema_trend_cross": _env_bool("STRAT_EMA_TREND_CROSS", True),
    "rsi_bollinger_reversion": _env_bool("STRAT_RSI_BOLLINGER", True),
    "fractal_breakout": _env_bool("STRAT_FRACTAL_BREAKOUT", True),
    "vwap_momentum": _env_bool("STRAT_VWAP_MOMENTUM", True),
}
CONFLUENCE_MIN_STRATEGIES = _env_int("CONFLUENCE_MIN_STRATEGIES", 2)  # de las habilitadas, minimo que deben coincidir
CONFLUENCE_MIN_AVG_CONFIDENCE = _env_float("CONFLUENCE_MIN_AVG_CONFIDENCE", 0.55)

# EMA trend cross
EMA_FAST = _env_int("EMA_FAST", 9)
EMA_SLOW = _env_int("EMA_SLOW", 21)
EMA_TREND = _env_int("EMA_TREND", 50)

# RSI + Bollinger
RSI_PERIOD = _env_int("RSI_PERIOD", 14)
RSI_OVERSOLD = _env_float("RSI_OVERSOLD", 30)
RSI_OVERBOUGHT = _env_float("RSI_OVERBOUGHT", 70)
BOLLINGER_PERIOD = _env_int("BOLLINGER_PERIOD", 20)
BOLLINGER_STD = _env_float("BOLLINGER_STD", 2.0)

# Fractal breakout
FRACTAL_WINDOW = _env_int("FRACTAL_WINDOW", 2)  # velas a cada lado para validar fractal (5 velas total)
BREAKOUT_MIN_ATR_MULTIPLE = _env_float("BREAKOUT_MIN_ATR_MULTIPLE", 0.3)  # ruptura minima para considerarse valida

# VWAP + momentum
MACD_FAST = _env_int("MACD_FAST", 12)
MACD_SLOW = _env_int("MACD_SLOW", 26)
MACD_SIGNAL = _env_int("MACD_SIGNAL", 9)
VWAP_SESSION_RESET_HOUR_UTC = _env_int("VWAP_SESSION_RESET_HOUR_UTC", 0)

# ---------------------------------------------------------------------------
# Frecuencia de trading
# ---------------------------------------------------------------------------
MIN_TRADES_PER_DAY_TARGET = _env_int("MIN_TRADES_PER_DAY_TARGET", 20)
MAX_TRADES_PER_DAY = _env_int("MAX_TRADES_PER_DAY", 400)
LOOP_INTERVAL_SEC = _env_float("LOOP_INTERVAL_SEC", 2.0)  # cadencia del loop principal (no confundir con cierre de vela M1)
COOLDOWN_AFTER_TRADE_SEC = _env_float("COOLDOWN_AFTER_TRADE_SEC", 5.0)

# ---------------------------------------------------------------------------
# AI optimizer (opcional, OpenRouter)
# ---------------------------------------------------------------------------
AI_OPTIMIZER_ENABLED = _env_bool("AI_OPTIMIZER_ENABLED", False)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
AI_OPTIMIZER_REVIEW_EVERY_N_TRADES = _env_int("AI_OPTIMIZER_REVIEW_EVERY_N_TRADES", 30)
AI_OPTIMIZER_MAX_PARAM_SHIFT_PCT = _env_float("AI_OPTIMIZER_MAX_PARAM_SHIFT_PCT", 15.0)  # clamp de seguridad

# ---------------------------------------------------------------------------
# Logging / rutas
# ---------------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
TRADES_LOG_PATH = LOG_DIR / "trades.csv"
STATE_PATH = LOG_DIR / "state.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DRY_RUN = _env_bool("DRY_RUN", True)  # True = calcula y loguea señales sin enviar ordenes reales
