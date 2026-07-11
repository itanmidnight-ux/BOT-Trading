#!/usr/bin/env python3
"""
BOT-Trading v4.1 — MetaTrader 5 / FBS (Windows / Linux)
Punto de entrada principal.
Uso:
  python main.py           -> menu interactivo
  python main.py --test    -> backtest/entrenamiento sin MT5 (modo test)
"""
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Flag global: True cuando se pasa --test (sin MT5 real)
_TEST_MODE = "--test" in sys.argv

# En modo test mockeamos MT5 ANTES de cualquier import del bot
if _TEST_MODE:
    from unittest.mock import MagicMock
    sys.modules["MetaTrader5"] = MagicMock()

from dotenv import load_dotenv
load_dotenv()

from utils.display import print_banner, print_menu
from utils.logger import get_logger
from config import settings, constants

_log = get_logger("main")

_IS_WINDOWS = platform.system() == "Windows"
_INSTALL_SCRIPT = "install.bat" if _IS_WINDOWS else "install.sh"
_VENV_ACTIVATE = r".venv\Scripts\activate" if _IS_WINDOWS else "source .venv/bin/activate"


def _check_prerequisites(require_mt5: bool = True) -> bool:
    """Verifica entorno. Si require_mt5=False omite check de librería MT5."""
    if not Path(".env").exists():
        print(f"\n[ERROR] .env no encontrado. Ejecuta {_INSTALL_SCRIPT}\n")
        return False
    if not Path(".venv").exists():
        print(f"\n[ERROR] .venv no encontrado. Ejecuta {_INSTALL_SCRIPT}\n")
        return False
    if require_mt5:
        try:
            from core.mt5_compat import mt5
        except Exception:
            print("\n[ERROR] MetaTrader5 no disponible.")
            print(f"  Activa venv: {_VENV_ACTIVATE}")
            if _IS_WINDOWS:
                print("  Luego: pip install MetaTrader5\n")
            else:
                print("  Luego: pip install mt5linux")
                print("  Y verifica que el bridge Wine (README) esté corriendo.\n")
            return False
    return True


def _connect_mt5(mode: str) -> bool:
    """Inicializa conexión MT5."""
    from core.mt5_connector import MT5Connector
    connector = MT5Connector()
    if not connector.initialize():
        print("\n[ERROR] No se pudo conectar a MT5.")
        print("  Asegúrate de que MT5 esté abierto y con sesión activa.")
        print(f"  Abre MetaTrader 5 manualmente e inicia sesión FBS-Demo.")
        return False

    account = connector.get_account_info()
    print(f"\n  [OK] Conectado: cuenta {account.get('login')} | "
          f"Balance: {account.get('balance', 0):.2f} | "
          f"Servidor: {account.get('server')}")
    return True


def run_backtest_only(symbol: str):
    """Flujo solo backtest sin conexión live."""
    print(f"\n[BACKTEST] Modo backtest para {symbol}")

    from core.data_downloader import download_historical, load_local
    from core.feature_engine import FeatureEngine
    from core.model_trainer import ModelTrainer
    from core.model_evaluator import ModelEvaluator
    from core.training_loop import TrainingLoop
    from core.backtester import Backtester
    from core.regime_detector import RegimeDetector
    from core.ollama_advisor import OllamaAdvisor
    from core.objective_engine import ObjectiveEngine
    from analysis.performance_reporter import PerformanceReporter

    fe        = FeatureEngine()
    trainer   = ModelTrainer()
    evaluator = ModelEvaluator()
    regime    = RegimeDetector()
    reporter  = PerformanceReporter()

    try:
        advisor = OllamaAdvisor()
        obj_eng = ObjectiveEngine(ollama_advisor=advisor)
        print("  [LLM] OllamaAdvisor activo.")
    except Exception:
        obj_eng = ObjectiveEngine()
        print("  [LLM] Ollama no disponible — objetivos sin LLM.")

    # Carga datos: yfinance si no hay locales (modo test/backtest sin MT5)
    df = load_local(symbol, settings.TIMEFRAME_MAIN)
    if df is None or len(df) < 1000:
        print(f"  Sin datos locales — descargando de yfinance...")
        try:
            import yfinance as yf, pandas as pd
            ticker = "EURUSD=X" if "EUR" in symbol else "XAUUSD=X"
            df = yf.download(ticker, period="90d", interval="1h",
                             progress=False, auto_adjust=True)
            df = df.reset_index()
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            df = df.rename(columns={"datetime": "time", "volume": "tick_volume"})
            df["tick_volume"] = df.get("tick_volume", 500)
            df["spread"] = 10
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df[["time","open","high","low","close","tick_volume","spread"]].dropna()
            print(f"  {len(df)} velas descargadas de yfinance")
        except Exception as e:
            print(f"\n[ERROR] No hay datos ni locales ni yfinance: {e}")
            return

    print(f"  Datos cargados: {len(df)} velas")

    # Training loop con gate 59% WR
    loop = TrainingLoop(symbol, fe, trainer, evaluator, Backtester, regime)
    result = loop.run(df, initial_capital=settings.INITIAL_CAPITAL, verbose=True)

    reporter.show(result, symbol)

    obj_result = obj_eng.evaluate(
        win_rate=result.get("win_rate", 0),
        profit_factor=result.get("profit_factor", 0),
        max_dd=result.get("max_drawdown", 0),
        n_trades=result.get("total_trades", 0),
    )
    obj = obj_eng.get_current()
    print(f"\n  [OBJETIVO] {obj.name} | "
          f"{'✓ LOGRADO' if obj_result.get('achieved') else '→ en progreso'}")
    if obj_result.get("achieved"):
        obj_eng.advance()

    if result.get("ready_for_live"):
        print(f"\n  [OK] Modelo listo para live. WR: {result['win_rate']:.1%}")
    else:
        print(f"\n  [WARN] WR {result['win_rate']:.1%} < 59% — no recomendado para live.")


def run_live(symbol: str, mode: str):
    """Flujo completo: datos → entrenamiento → backtest → live."""
    from core.mt5_connector import MT5Connector
    from core.mt5_stream import MT5Stream
    from core.data_downloader import download_historical
    from core.data_updater import DataUpdater
    from core.feature_engine import FeatureEngine
    from core.model_trainer import ModelTrainer
    from core.model_evaluator import ModelEvaluator
    from core.model_updater import ModelUpdater
    from core.training_loop import TrainingLoop
    from core.backtester import Backtester
    from core.regime_detector import RegimeDetector
    from core.multi_tf_analyzer import MultiTFAnalyzer
    from core.signal_generator import SignalGenerator
    from core.kelly import KellyEngine
    from core.risk_manager import RiskManager
    from core.capital_scaler import CapitalScaler
    from core.trade_manager import TradeManager
    from core.exit_manager import ExitManager
    from core.state_manager import StateManager
    from core.auto_improver import AutoImprover
    from core.live_trader import LiveTrader
    from core.ollama_advisor import OllamaAdvisor
    from core.objective_engine import ObjectiveEngine
    from analysis.performance_reporter import PerformanceReporter

    # LLM + objetivos autónomos
    try:
        advisor  = OllamaAdvisor()
        obj_eng  = ObjectiveEngine(ollama_advisor=advisor)
        _log.info("OllamaAdvisor + ObjectiveEngine activos.")
    except Exception as e:
        _log.warning("Ollama no disponible (%s) — modo sin LLM.", e)
        advisor  = None
        obj_eng  = ObjectiveEngine()

    connector = MT5Connector()
    account   = connector.get_account_info()
    equity    = account.get("equity", settings.INITIAL_CAPITAL)

    # ── Fase 1: Descarga de datos ─────────────────────────────────────────────
    print(f"\n[1/4] Descargando datos históricos {symbol}...")
    df_m1 = download_historical(symbol, "M1", n_candles=settings.MIN_CANDLES_TO_TRAIN)

    if df_m1 is None or len(df_m1) < 10000:
        print(f"[ERROR] Datos insuficientes: {len(df_m1) if df_m1 is not None else 0} velas")
        return

    print(f"  {len(df_m1)} velas M1 disponibles")

    # ── Fase 2: Training loop con gate 59% WR ────────────────────────────────
    print(f"\n[2/4] Entrenamiento iterativo (objetivo WR >= 59%)...")
    fe        = FeatureEngine()
    trainer   = ModelTrainer()
    evaluator = ModelEvaluator()
    regime    = RegimeDetector()

    loop   = TrainingLoop(symbol, fe, trainer, evaluator, Backtester, regime)
    result = loop.run(df_m1, initial_capital=equity, verbose=True)

    PerformanceReporter().show(result, symbol)

    if not result.get("ready_for_live"):
        wr = result.get("win_rate", 0)
        print(f"\n[WARN] El modelo alcanzó WR={wr:.1%} (mínimo 59%).")
        resp = input("  ¿Continuar de todas formas? (escribe SI para confirmar): ").strip().upper()
        if resp != "SI":
            print("  Saliendo. Revisa los datos o ajusta parámetros.")
            return

    # ── Fase 3: Configurar componentes live ──────────────────────────────────
    print(f"\n[3/4] Inicializando componentes de trading...")
    model_upd   = ModelUpdater(symbol)
    mtf         = MultiTFAnalyzer()
    state       = StateManager()
    trade_mgr   = TradeManager()
    exit_mgr    = ExitManager()
    kelly       = KellyEngine(symbol)
    risk        = RiskManager(equity)
    scaler      = CapitalScaler()
    data_upd    = DataUpdater([symbol], settings.ALL_TIMEFRAMES)
    improver    = AutoImprover()
    stream      = MT5Stream()

    signal_gen = SignalGenerator(model_upd, regime, mtf, state)

    live = LiveTrader(
        symbol=symbol, mt5_connector=connector, mt5_stream=stream,
        feature_engine=fe, model_updater=model_upd, regime_detector=regime,
        mtf_analyzer=mtf, signal_generator=signal_gen, kelly_engine=kelly,
        risk_manager=risk, trade_manager=trade_mgr, exit_manager=exit_mgr,
        state_manager=state, data_updater=data_upd, capital_scaler=scaler,
        auto_improver=improver,
    )

    # Evaluar objetivo con resultado del training
    obj_result = obj_eng.evaluate(
        win_rate=result.get("win_rate", 0),
        profit_factor=result.get("profit_factor", 0),
        max_dd=result.get("max_drawdown", 0),
        n_trades=result.get("total_trades", 0),
    )
    obj = obj_eng.get_current()
    print(f"\n  [OBJETIVO] {obj.name} | "
          f"{'✓ LOGRADO' if obj_result.get('achieved') else '→ en progreso'}")
    if obj_result.get("achieved"):
        obj_eng.advance()

    # ── Fase 4: Live trading ──────────────────────────────────────────────────
    print(f"\n[4/4] Iniciando trading en vivo — {mode.upper()} — {symbol}")
    print(f"  Balance: ${equity:.2f} | Fase: {scaler.get_phase(equity)} | WR modelo: {result['win_rate']:.1%}")
    live.start()


def run_log_analysis():
    """Analiza todos los logs y muestra resumen."""
    from analysis.log_analyzer import LogAnalyzer
    analyzer = LogAnalyzer()
    for symbol in [settings.SYMBOL_PHASE1, settings.SYMBOL_PHASE2]:
        result = analyzer.analyze(symbol)
        if result.get("summary"):
            s = result["summary"]
            print(f"\n  {symbol}: {s['total_trades']} trades | WR: {s['win_rate']:.1%} | "
                  f"PF: {s['profit_factor']:.2f} | PnL: ${s['net_pnl']:+.2f}")


def run_optimize(symbol: str):
    """Optimiza parámetros con grid search."""
    from core.data_downloader import load_local
    from core.feature_engine import FeatureEngine
    from core.model_trainer import ModelTrainer
    from core.backtester import Backtester
    from analysis.optimizer import Optimizer

    df = load_local(symbol, settings.TIMEFRAME_MAIN)
    if df is None:
        print("[ERROR] No hay datos locales. Descarga primero.")
        return

    fe      = FeatureEngine()
    trainer = ModelTrainer()
    df_feat = fe.compute(df)
    df_feat = fe.add_target(df_feat, symbol)
    df_feat = df_feat.dropna()
    feature_cols = fe.get_feature_cols()

    metrics = trainer.train(df_feat, symbol)
    model   = trainer.last_model

    opt    = Optimizer()
    result = opt.optimize(symbol, df_feat, feature_cols, model, Backtester)
    if result["best_params"]:
        print(f"\n  Mejores parámetros: {result['best_params']}")
        print(f"  PF validación: {result['best_pf']:.2f}")
    else:
        print("\n  No se encontraron parámetros mejores.")


def main():
    print_banner()

    # Modo test: sin MT5, backtest directo con yfinance
    if _TEST_MODE:
        print("\n  [TEST MODE] Sin MT5 — backtest + LLM con datos yfinance\n")
        if not _check_prerequisites(require_mt5=False):
            sys.exit(1)
        run_backtest_only(settings.SYMBOL_PHASE1)
        return

    if not _check_prerequisites(require_mt5=True):
        sys.exit(1)

    while True:
        print_menu()
        choice = input("\n  Tu elección: ").strip()

        if choice == "1":
            print("\n  [DEMO] Modo demo (cuenta demo FBS)")
            if _connect_mt5("demo"):
                run_live(settings.SYMBOL_PHASE1, "demo")

        elif choice == "2":
            print("\n  [REAL] Modo real — DINERO REAL")
            confirm = input("  Escribe CONFIRMO para continuar: ").strip().upper()
            if confirm == "CONFIRMO":
                if _connect_mt5("real"):
                    run_live(settings.SYMBOL_PHASE1, "real")
            else:
                print("  Cancelado.")

        elif choice == "3":
            print("\n  [BACKTEST] Modo backtest sin conexión live")
            run_backtest_only(settings.SYMBOL_PHASE1)

        elif choice == "4":
            print("\n  [ANÁLISIS] Analizando logs...")
            run_log_analysis()

        elif choice == "5":
            print("\n  [OPTIMIZAR] Optimizando parámetros...")
            run_optimize(settings.SYMBOL_PHASE1)

        elif choice == "6":
            print("\n  Hasta pronto.\n")
            break

        else:
            print("  Opción no válida.")


if __name__ == "__main__":
    main()
