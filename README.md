# BOT-Trading v4.1 — MT5/FBS Scalping Bot

Bot de trading algorítmico para MetaTrader 5 con broker FBS. Usa Ensemble ML, Kelly Criterion y un LLM local (Ollama) para auto-mejora autónoma.

---

## ¿Qué hace?

1. **Descarga datos** históricos de MT5 (EUR/USD M1)
2. **Genera 57 features** técnicos (Heikin-Ashi, VWAP, Fisher, orderflow, session)
3. **Entrena un Ensemble** XGBoost + LightGBM + CatBoost + MLP con votación ponderada
4. **Gate de calidad**: solo pasa al live si el backtest alcanza ≥ 59% Win Rate
5. **Opera en vivo** con Kelly Criterion, trailing SL, TP parcial 50% y break-even
6. **LLM autónomo** (Ollama) evalúa resultados y sugiere mejoras automáticamente

---

## Arquitectura

```
MT5 → data_downloader → feature_engine → ensemble_model
    → training_loop (gate 59% WR) → signal_generator
    → kelly → trade_manager → exit_manager → live_trader
                    ↕
         ollama_advisor + objective_engine (LLM)
```

### Módulos principales

| Módulo | Función |
|--------|---------|
| `core/training_loop.py` | Entrena iterativamente hasta alcanzar 59% WR (máx 20 iteraciones) |
| `core/ensemble_model.py` | XGBoost + LightGBM + CatBoost + MLP con votación ponderada |
| `core/advanced_features.py` | 57 features: Heikin-Ashi, VWAP, choppiness, Fisher, orderflow |
| `core/kelly.py` | Kelly Criterion con leverage 1:3000 (EUR/USD) |
| `core/exit_manager.py` | Trailing SL, TP parcial 50%, break-even, time exit (30 velas) |
| `core/ollama_advisor.py` | LLM local para análisis y sugerencias autónomas |
| `core/objective_engine.py` | 7 objetivos autónomos: Bootstrap → Elite (59% → 75% WR) |
| `core/rl_overlay.py` | Q-Learning para ajuste dinámico de threshold (216 estados) |
| `core/state_manager.py` | Persiste estado en `logs/system/state.json` |
| `core/capital_scaler.py` | Fase 1 EUR/USD → Fase 2 XAU/USD cuando capital ≥ $30 |

---

## Requisitos

- **Windows 10/11** (64-bit)
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **MetaTrader 5** — [descargar aquí](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe)
- **Cuenta demo FBS** — [fbs.com](https://fbs.com)
- **Ollama** (opcional, para LLM) — [ollama.ai](https://ollama.ai)

---

## Instalación

### 1. Clonar el repositorio

```bat
git clone https://github.com/itanmidnight-ux/BOT-Trading.git
cd BOT-Trading
```

### 2. Instalar todo automáticamente

```bat
install.bat
```

Esto instala: virtualenv Python, dependencias, verifica MT5, descarga modelo Ollama (phi3:mini).

### 3. Instalar la librería MetaTrader5

```bat
.venv\Scripts\activate
pip install MetaTrader5
```

### 4. Configurar credenciales

Edita el archivo `.env`:

```env
TRADING_MODE=demo
MT5_EXE=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=TU_LOGIN
MT5_PASSWORD=TU_PASSWORD
MT5_SERVER=FBS-Demo
```

---

## Uso

### Modo TEST (sin MT5, sin cuenta real)

Entrena el modelo con datos de yfinance y hace backtest. **No necesita MT5 instalado.**

```bat
run_test.bat
```

o desde terminal:

```bat
.venv\Scripts\activate
python main.py --test
```

### Modo BACKTEST (con datos MT5 locales)

```bat
.venv\Scripts\activate
python main.py
# Seleccionar opción 3
```

### Modo DEMO (trading real con cuenta demo)

1. Abre MetaTrader 5
2. Inicia sesión en tu cuenta FBS Demo
3. Deja MT5 abierto en segundo plano
4. Ejecuta:

```bat
.venv\Scripts\activate
python main.py
# Seleccionar opción 1
```

### Menú principal

```
[1] Demo    — Trading en vivo con cuenta demo FBS
[2] Real    — Trading con dinero real (requiere confirmación)
[3] Backtest — Entrena + backtesta sin abrir trades
[4] Análisis — Analiza logs históricos
[5] Optimizar — Optimiza parámetros automáticamente
[6] Salir
```

---

## Parámetros clave

| Parámetro | Valor |
|-----------|-------|
| Win Rate mínimo para live | 59% |
| Win Rate objetivo | 70%+ |
| Kelly máximo | 20% del capital |
| Max daily loss | 5% |
| Max drawdown | 15% |
| Leverage EUR/USD | 1:3000 |
| Leverage XAU/USD | 1:1000 |
| Fase 2 (XAU/USD) | Capital ≥ $30 |
| Timeframe principal | M1 |
| Time exit | 30 velas |

---

## Flujo de entrenamiento iterativo

```
Datos M1 → Features (57) → Split temporal (sin data leakage)
    ↓
Iteración 1..20:
    Entrena Ensemble → Backtest → WR >= 59%?
        SI → Guarda modelo → Pasa a live
        NO → Ajusta SIGNAL_THRESHOLD / ATR_SL / ATR_TP → Siguiente iteración
    ↓
LLM evalúa resultado → Genera objetivo siguiente
```

---

## Estructura del proyecto

```
BOT-Trading/
├── main.py                 # Punto de entrada
├── install.bat             # Instalador Windows
├── run_test.bat            # Modo test sin MT5
├── start_training.bat      # Entrenamiento offline
├── requirements.txt
├── .env                    # Credenciales (NO subir a git)
├── config/
│   ├── settings.py         # Configuración principal
│   ├── constants.py        # Constantes del sistema
│   └── .env.example        # Plantilla de configuración
├── core/                   # Módulos del bot (12 en pipeline)
├── analysis/               # Reportes y optimización
├── utils/                  # Logger, display, notifier
├── tests/                  # 40 tests automatizados
├── logs/                   # Trades, backtest, sistema
├── data/                   # Datos históricos
└── models/                 # Checkpoints y reportes
```

---

## Tests

```bat
.venv\Scripts\activate
python -m pytest tests/ -v
```

**40/40 tests pasando.** El pipeline completo se testea con datos reales de yfinance (mock de MT5).

---

## Objetivos autónomos (LLM)

El `ObjectiveEngine` sigue una escalera de 7 objetivos:

| Nivel | Nombre | WR objetivo |
|-------|--------|-------------|
| 1 | Bootstrap | 59% |
| 2 | Stable | 62% |
| 3 | Consistent | 65% |
| 4 | Profitable | 67% |
| 5 | Advanced | 70% |
| 6 | Expert | 72% |
| 7 | Elite | 75% |

Una vez completados los 7, el LLM (Ollama) genera objetivos personalizados automáticamente.

---

## Notas de seguridad

- El archivo `.env` está en `.gitignore` — **nunca subas tus credenciales**
- Usa siempre cuenta **demo** para pruebas
- El modo **real** requiere escribir `CONFIRMO` para activarse
- Max drawdown del 15% desactiva el bot automáticamente

---

## Licencia

Uso personal. No usar en producción sin entender completamente el riesgo de trading.
