# Bot XAUUSD 1m

Bot de trading algoritmico para XAUUSD en timeframe M1: orquestador en Python
(confluencia de 4 estrategias, gestion de riesgo, grid de contencion, SL/TP
dinamicos por ATR, trailing con cierre en maxima ganancia) + un Expert Advisor
en MQL5 (`expert_advisor.mq5`) como via de ejecucion alternativa dentro de MT5.

## ⚠️ Antes de usar esto con dinero real

- **Ningun bot puede garantizar ganancias.** Este repositorio implementa la
  arquitectura, la logica de riesgo y las estrategias solicitadas, pero no
  trae backtests con resultados reales embebidos ni ninguna promesa de
  rentabilidad — eso depende del broker, el spread real, las condiciones de
  mercado y de que vos mismo corras la validacion descrita mas abajo.
- **`expert_advisor.mq5` no fue compilado ni testeado en un terminal MT5
  real** al generarse este codigo (el entorno de desarrollo no tenia
  MetaEditor/MT5 disponible). Abrilo en MetaEditor, compila (F7), revisa que
  no haya errores/warnings, y corre el Strategy Tester antes de usarlo.
- **El backtester (`backtester.py`) no incluye datos historicos.** Tenes que
  proveerle vos velas M1 reales (6+ meses recomendado) exportadas de tu
  broker antes de que sus metricas signifiquen algo.
- `DRY_RUN=True` es el default: calcula y loguea todo (señales, SL/TP,
  tamaño de posicion) sin enviar ninguna orden real. Dejalo asi hasta validar
  el comportamiento en una cuenta demo.
- La frecuencia objetivo de 20-400 trades/dia es agresiva para 1m: a mayor
  frecuencia, mayor peso relativo tienen spread/comision/slippage sobre el
  resultado neto. `MAX_TRADES_PER_DAY` es un techo de seguridad, no una meta
  que el sistema fuerce artificialmente — la confluencia y los gates de
  riesgo priman siempre sobre la frecuencia.

## Arquitectura

```
                 ┌─────────────────────────────┐
                 │           main.py            │  orquestador (loop principal)
                 └──────────────┬───────────────┘
                                 │
   ┌───────────┬─────────────┬──┴──────────┬──────────────┬───────────────┐
   │            │             │             │              │               │
capital_    strategies/  stop_loss_   risk_management  grid_trader   profit_manager
detector    confluence_  calculator                                  py
.py         engine.py    .py

                 │
        mt5_connector/  (connector.py = API directa | bridge.py = EA)
                 │
        mt5_compat.py (MetaTrader5 nativo o bridge mt5linux en Linux/Wine)
                 │
           expert_advisor.mq5  (via alternativa de ejecucion, USE_EA_BRIDGE=True)
```

### Flujo por ciclo (`main.py`)
1. `capital_detector.py` lee balance/equity/margen libre reales de la cuenta.
2. Se descargan velas M1 recientes y se calcula ATR.
3. Se sincronizan posiciones abiertas reales contra el estado interno.
4. `profit_manager.py` gestiona cada posicion abierta: TP escalonado (TP1/TP2
   parciales), trailing dinamico armado por ATR, y cierre total cuando el
   profit flotante retrocede mas de `PROFIT_GIVEBACK_TOLERANCE_PCT` desde su
   pico — esa es la implementacion concreta de "cerrar en la maxima ganancia".
5. Si hay cupo de riesgo (`risk_management.py`: perdida diaria, drawdown,
   maximo de posiciones, maximo de trades/dia) y las 4 estrategias tienen
   confluencia (`strategies/confluence_engine.py`), se calcula el plan de
   SL/TP con ATR (`stop_loss_calculator.py`), se dimensiona el volumen segun
   el capital real detectado, y se abre la posicion.
6. Si `GRID_ENABLED=True`, cada posicion base arma una sesion de grid
   (`grid_trader.py`) con niveles de escalado acotados por
   `GRID_MAX_TOTAL_RISK_PCT` — la red de contencion, no una martingala sin
   limite.

## Estrategias (confluencia, no señales aisladas)

Un trade solo se abre cuando al menos `CONFLUENCE_MIN_STRATEGIES` de las
estrategias habilitadas coinciden en direccion, con confianza promedio ≥
`CONFLUENCE_MIN_AVG_CONFIDENCE` (ver `strategies/confluence_engine.py`).

| Estrategia | Archivo | Logica |
|---|---|---|
| EMA trend cross | `strategies/ema_trend_cross.py` | Cruce EMA9/EMA21 filtrado por EMA50 de tendencia |
| RSI + Bollinger reversion | `strategies/rsi_bollinger_reversion.py` | Reversion en banda extrema + RSI en zona + vela de rechazo |
| Fractal breakout | `strategies/fractal_breakout.py` | Ruptura de soporte/resistencia (fractales de Williams) con colchon ATR |
| VWAP + momentum | `strategies/vwap_momentum.py` | Precio vs VWAP intradia + histograma MACD confirmando |

Cada modulo documenta su logica de entrada/salida y calculo de confianza en
su propio docstring.

## Stop loss (prioridad critica #1)

`stop_loss_calculator.py` documenta el calculo paso a paso: distancia base
por ATR, distancia minima del broker (`trade_stops_level` + colchon
configurable), validacion de spread, y verificacion de RR minimo antes de
considerar el plan valido. Ver el docstring del modulo para el detalle
matematico completo.

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completar credenciales
```

En Linux, si el terminal MT5 corre bajo Wine, levanta el servidor `mt5linux`
con el Python de Wine y configura `MT5_BRIDGE_HOST`/`MT5_BRIDGE_PORT` en
`.env`; `mt5_compat.py` detecta automaticamente ese backend.

## Uso

```bash
# Validar logica sin enviar ordenes reales (default DRY_RUN=True)
python main.py

# Backtest con tus propios datos historicos (CSV: time,open,high,low,close,tick_volume)
python backtester.py ruta/a/tus_datos_xauusd_m1.csv

# Tests unitarios (no requieren MT5 ni conexion de red)
pytest tests/ -v
```

### Habilitar ordenes reales
1. Corre en demo con `DRY_RUN=True` primero y revisa `logs/` (señales,
   tamaños de posicion, SL/TP calculados) durante varios dias.
2. Cambia `DRY_RUN=False` en `.env` **solo en cuenta demo** al principio.
3. Corre `expert_advisor.mq5` en el Strategy Tester de MT5 y revisa que
   compile y opere sin errores antes de considerar `USE_EA_BRIDGE=True`.
4. Solo despues de validar en demo, pasar a cuenta real con capital que
   puedas permitirte perder — los limites de `MAX_DAILY_LOSS_PCT` y
   `MAX_DRAWDOWN_PCT` en `config.py` son la ultima red de seguridad, no un
   sustituto de la validacion previa.

## Configuracion

Todos los parametros (riesgo, ATR, grid, trailing, confluencia, frecuencia de
trading, AI optimizer) estan centralizados en `config.py`, sobreescribibles
por variable de entorno. Ver los comentarios de cada seccion en ese archivo.

## AI optimizer (opcional)

`ai_optimizer.py` puede consultar un modelo gratuito de OpenRouter para
sugerir ajustes de parametros no criticos (confianza minima de confluencia,
multiplicadores de ATR, tolerancia de giveback) en base al historial reciente
de trades. Es puramente asesor: escribe sus sugerencias en
`logs/ai_suggested_params.json` para revision humana y **nunca las aplica
automaticamente** — no toca limites de riesgo duros (perdida diaria,
drawdown, maximo de posiciones). Deshabilitado por default
(`AI_OPTIMIZER_ENABLED=False`).

## Estructura del repositorio

```
main.py                    orquestador principal
config.py                  parametros centralizados
capital_detector.py        deteccion automatica de capital/equity
indicators.py               EMA, RSI, ATR, Bollinger, MACD, VWAP, fractales
strategies/
  base.py                  tipos comunes (Signal, Direction)
  ema_trend_cross.py
  rsi_bollinger_reversion.py
  fractal_breakout.py
  vwap_momentum.py
  confluence_engine.py     combina las 4 estrategias
mt5_connector/
  connector.py             API MetaTrader5 directa (via primaria)
  bridge.py                bridge de archivos hacia expert_advisor.mq5 (via alternativa)
mt5_compat.py               resuelve MetaTrader5 nativo / mt5linux / stub
risk_management.py         position sizing, limites de perdida/drawdown
grid_trader.py              grid de contencion de perdidas
stop_loss_calculator.py    SL/TP dinamicos por ATR
profit_manager.py          trailing, TP escalonado, cierre en maxima ganancia
backtester.py               motor de backtesting (requiere datos propios)
ai_optimizer.py             sugerencias opcionales via OpenRouter
expert_advisor.mq5          EA puente (via alternativa de ejecucion)
tests/                       tests unitarios sin dependencia de MT5
```
