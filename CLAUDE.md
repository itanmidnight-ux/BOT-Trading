# CLAUDE.md — v4.1 (Windows)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos principales (Windows)

```bat
# Instalar todo (primera vez)
install.bat

# Correr tests
.venv\Scripts\activate && python -m pytest tests/ -v

# Entrenar sin MT5
start_training.bat

# Iniciar bot (MT5 debe estar abierto y con sesión activa)
.venv\Scripts\activate && python main.py
```

## Arquitectura

Pipeline secuencial de 12 módulos en `core/`:

```
MT5 → data_downloader → feature_engine → model_trainer
    → training_loop (59% WR gate) → signal_generator
    → kelly → trade_manager → exit_manager → live_trader
```

### Flujo de entrenamiento iterativo (crítico)
`core/training_loop.py` implementa el gate de 59% WR:
- Entrena XGBoost → backtesta → si WR < 59%, ajusta parámetros y re-entrena
- Máximo 20 iteraciones
- Ajusta: SIGNAL_THRESHOLD, ATR_SL_MULTIPLIER, ATR_TP1_MULTIPLIER
- Guarda mejores parámetros en `config/runtime_params.json`

### Módulos clave
- `config/settings.py`: carga `runtime_params.json` + `optimized_params.json` con prioridad sobre defaults
- `core/exit_manager.py`: trailing SL, TP parcial 50%, break-even, time exit (30 velas)
- `core/kelly.py`: Kelly Criterion con leverage 1:3000 (EUR_USD) / 1:1000 (XAU_USD)
- `core/capital_scaler.py`: Fase 1 EUR_USD → Fase 2 USD cuando capital >= $30
- `core/state_manager.py`: persiste estado en `logs/system/state.json`, verifica contra MT5 al arrancar

### MT5 en Windows
MT5 corre nativamente en Windows. `mt5_connector.py` lo lanza via `subprocess.Popen` si no está abierto. Credenciales en `.env` (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).

## Sistemas v4.0 añadidos
- `core/ensemble_model.py` — XGBoost + LightGBM + CatBoost votación ponderada
- `core/advanced_features.py` — 57 features (28 base + 29 avanzados: Heikin-Ashi, VWAP, choppiness, Fisher, orderflow, session)
- `core/lstm_model.py` — MLP secuencial (4to votante ensemble)
- `core/rl_overlay.py` — Q-Learning ajuste dinámico de threshold (216 estados)
- `core/ollama_advisor.py` — LLM local para auto-mejora autónoma
- `core/objective_engine.py` — 7 objetivos autónomos: Bootstrap → Elite (59%→75% WR)

## Parámetros importantes
- WR mínimo para live: 59% (`MIN_WIN_RATE_LIVE`)
- Objetivo: 70%+ WR (`TARGET_WIN_RATE`)
- Kelly max: 20% del capital
- Max daily loss: 5%
- Max drawdown: 15%
- Fase 2 (XAU_USD) activa cuando capital >= $30
