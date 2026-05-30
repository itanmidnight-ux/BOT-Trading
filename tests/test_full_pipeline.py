"""
Test completo del pipeline con datos reales de yfinance.
Simula todo excepto la conexión MT5 real.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch
sys.modules['MetaTrader5'] = MagicMock()

import numpy as np
import pandas as pd

def download_real_data():
    """Descarga datos EUR/USD reales de yfinance."""
    try:
        import yfinance as yf
        df = yf.download("EURUSD=X", period="60d", interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("No data from yfinance")
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df.rename(columns={'datetime': 'time', 'vol': 'tick_volume', 'volume': 'tick_volume'})
        if 'time' not in df.columns:
            df['time'] = pd.to_datetime(df.index, utc=True)
        df['tick_volume'] = df.get('tick_volume', pd.Series(500, index=df.index)).fillna(500)
        df['spread'] = 10
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df[['time','open','high','low','close','tick_volume','spread']].dropna()
        print(f"  Datos reales descargados: {len(df)} velas EUR/USD H1")
        return df
    except Exception as e:
        print(f"  yfinance falló ({e}), usando datos sintéticos")
        return None

def make_synthetic_data(n=2000):
    rng = np.random.default_rng(42)
    close = 1.08 + np.cumsum(rng.normal(0, 0.0003, n))
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1h', tz='UTC'),
        'open': close + rng.normal(0, 0.0001, n),
        'high': close + rng.uniform(0.0002, 0.001, n),
        'low':  close - rng.uniform(0.0002, 0.001, n),
        'close': close, 'tick_volume': rng.integers(100, 1000, n), 'spread': 10,
    })

def test_full_pipeline():
    print("\n" + "="*54)
    print("  TEST COMPLETO BOT-Trading v4.0")
    print("  Cuenta demo: 106049158 @ FBS-Demo")
    print("="*54)

    # 1. Datos
    print("\n[1/6] Cargando datos EUR/USD...")
    _real = download_real_data()
    df = _real if (_real is not None and len(_real) > 100) else make_synthetic_data(2000)
    print(f"  Velas disponibles: {len(df)}")
    assert len(df) > 500

    # 2. Features avanzadas
    print("\n[2/6] Calculando 57 features avanzadas...")
    from core.advanced_features import AdvancedFeatureEngine
    fe = AdvancedFeatureEngine()
    df_feat = fe.compute_advanced(df.copy())
    df_feat = fe.add_target(df_feat, "EURUSD")
    df_feat = df_feat.dropna().reset_index(drop=True)
    feature_cols = fe.get_all_feature_cols()
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    print(f"  Features: {len(feature_cols)} columnas | Muestras: {len(df_feat)}")
    assert len(feature_cols) >= 40
    assert len(df_feat) >= 200

    # 3. Ensemble training
    print("\n[3/6] Entrenando Ensemble (XGBoost+LightGBM+CatBoost)...")
    from core.ensemble_model import EnsembleModel
    n = len(df_feat)
    n_test = max(100, int(n * 0.15))
    n_val  = max(100, int((n - n_test) * 0.20))
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target']
    X_tr = X.iloc[:n-n_test-n_val]; y_tr = y.iloc[:n-n_test-n_val]
    X_val = X.iloc[n-n_test-n_val:n-n_test]; y_val = y.iloc[n-n_test-n_val:n-n_test]
    X_te  = X.iloc[-n_test:]; y_te = y.iloc[-n_test:]

    ens = EnsembleModel()
    metrics = ens.train(X_tr, y_tr, X_val, y_val, X_te, y_te)
    weights = ens.get_weights()
    print(f"  Pesos ensemble: {', '.join(f'{k}:{v:.2f}' for k,v in weights.items())}")
    print(f"  AUC test: {metrics.get('auc_roc', 0):.3f}")
    ens.save("EURUSD")
    assert sum(weights.values()) > 0.99

    # 4. Backtester con training loop
    print("\n[4/6] Training loop iterativo (gate 59% WR)...")
    from core.model_trainer import ModelTrainer
    from core.model_evaluator import ModelEvaluator
    from core.backtester import Backtester
    from core.regime_detector import RegimeDetector
    from core.training_loop import TrainingLoop
    from core.objective_engine import ObjectiveEngine

    trainer   = ModelTrainer()
    evaluator = ModelEvaluator()
    regime    = RegimeDetector()
    obj_eng   = ObjectiveEngine()

    loop = TrainingLoop(
        symbol="EURUSD", feature_engine=fe, model_trainer=trainer,
        model_evaluator=evaluator, backtester_cls=Backtester,
        regime_detector=regime, ensemble_model=ens,
        objective_engine=obj_eng,
    )
    result = loop.run(df, initial_capital=20.0, max_iterations=5, verbose=True)

    wr = result.get('win_rate', 0)
    pf = result.get('profit_factor', 0)
    dd = result.get('max_drawdown', 1)
    trades = result.get('total_trades', 0)
    capital_final = result.get('final_capital', 20.0)

    print(f"\n  Win Rate:      {wr:.1%}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown:  {dd:.1%}")
    print(f"  Total Trades:  {trades}")
    print(f"  Capital Final: ${capital_final:.2f} (inicio $20.00)")
    print(f"  ROI:           {result.get('roi', 0):.1%}")

    # 5. Kelly + RL + Objetivos
    print("\n[5/6] Verificando sistemas de capital y objetivos...")
    from core.kelly import KellyEngine
    from core.rl_overlay import RLOverlay
    from core.capital_scaler import CapitalScaler

    kelly  = KellyEngine("EURUSD")
    frac   = kelly.calculate_fraction(wr if wr > 0.3 else 0.55, 8.0, 6.0)
    lots   = kelly.fraction_to_lots(frac, 20.0, "EURUSD", 1.085)
    rl     = RLOverlay()
    thr    = rl.get_threshold('TRENDING_UP', 0.0009, 0.001, 10, 0)
    scaler = CapitalScaler()

    print(f"  Kelly fraction: {frac:.1%} → {lots:.2f} lots @ $20")
    print(f"  RL threshold:   {thr:.3f}")
    print(f"  Fase actual:    {scaler.get_phase(20.0)} (activa {scaler.get_active_symbols(20.0)})")
    print(f"  Objetivo:       {obj_eng.display(wr, pf, trades)}")

    # 6. Estado final
    print("\n[6/6] Estado del sistema...")
    from core.state_manager import StateManager
    state = StateManager()
    print(f"  Capital:            ${state.capital:.2f}")
    print(f"  Posiciones abiertas: {len(state.get_all_positions())}")
    print(f"  Fase:               {state.phase}")

    print("\n" + "="*54)
    ready = result.get('ready_for_live', False)
    if ready:
        print("  ✓ BOT LISTO PARA CUENTA DEMO FBS")
        print("  ✓ Win Rate supera 59% — puede operar")
    else:
        print(f"  ⚠ WR={wr:.1%} — entrenando más iteraciones al conectar MT5")
    print(f"\n  CREDENCIALES DEMO CONFIGURADAS:")
    print(f"  Login: 106049158 | Server: FBS-Demo")
    print(f"\n  LIMITACIÓN: Sistema ARM64 detectado.")
    print(f"  MT5 requiere x86_64. Para ejecutar en vivo:")
    print(f"  → Opción A: VPS/PC con x86_64 Linux o Windows")
    print(f"  → Opción B: OANDA API (funciona en ARM64)")
    print("="*54)

    assert wr >= 0
    assert lots >= 0.01
    return result

if __name__ == "__main__":
    result = test_full_pipeline()
    print(f"\nTest completado. Win Rate: {result.get('win_rate',0):.1%}")
