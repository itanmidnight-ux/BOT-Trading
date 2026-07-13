# Diseño: XAUUSD scalping de 1 vela (M1, leverage 1:1)

## Contexto

El bot pasa de operar dos símbolos con fases (EUR_USD fase1 → XAU_USD fase2,
Kelly con leverage 1:3000/1:1000) a operar **exclusivamente XAUUSD en M1 con
apalancamiento 1:1**, ejecutando un trade en cada cierre de vela para
explotar el ruido intra-M1 y buscar ganancias pequeñas y consistentes.

## Investigación de parámetros (Fable 5)

Se delegó a un agente con modelo Fable 5 la búsqueda de parámetros óptimos
para el objetivo "mayoría de trades con ≥$0.30 neto, 1000+ trades/día".

Hallazgo: con spread real de XAUUSD (~$0.20–0.40/oz) y el gate de calidad
existente (≥59% WR), no existe combinación SL:TP que sea rentable neta de
spread en M1 — el spread iguala o supera la ganancia objetivo de $0.30, y a
capital $20 con leverage 1:1 no alcanza margen ni para el lote mínimo
estándar (0.01 lote = 1oz ≈ $3300 de margen necesario a 1:1). Se documenta
para trazabilidad; por instrucción explícita del usuario se implementa el
diseño igual, usando los parámetros "menos malos" que arrojó la
investigación. El comportamiento en vivo con capital insuficiente ya está
cubierto por la verificación de margen existente (`kelly.verify_margin`),
que rechaza/reduce el trade sin intervención adicional.

## Parámetros

- `SYMBOL = "XAUUSD"` (único símbolo, sin fases)
- `LEVERAGE_XAUUSD = 1`
- `ATR_SL_MULTIPLIER = 1.0`
- `ATR_TP_MULTIPLIER = 1.2` (reemplaza a ATR_TP1_MULTIPLIER; se elimina TP2)
- `KELLY_MIN_LOTS = 0.01`
- `MAX_SPREAD_USD = 0.35` (filtro: si spread instantáneo > este valor, se
  salta la vela — única excepción a "trade en cada vela")

## Cambios de arquitectura

1. **Símbolo único**: se elimina el sistema de fases (`capital_scaler.py`
   deja de invocarse desde `live_trader.py`/`main.py`). `SYMBOL_PHASE1`/
   `SYMBOL_PHASE2`/`CAPITAL_PHASE2_THRESHOLD`/`LEVERAGE_PHASE1` se retiran de
   `config/settings.py` y se reemplazan por `SYMBOL` y `LEVERAGE_XAUUSD`.

2. **Ciclo de 1 vela**: en `live_trader._on_bar_close`, si hay posición
   abierta se fuerza su cierre a mercado antes de evaluar la siguiente señal
   (ya no hay hold multi-vela). `exit_manager.evaluate` se simplifica: solo
   SL (protección) y TP (ganancia chica); se elimina TP2/trailing/partial/
   time-exit — el cierre por tiempo pasa a ser automático (fin de la vela).

3. **Señal siempre direccional**: `signal_generator.generate` deja de
   devolver `HOLD` por threshold/régimen — cada vela produce BUY o SELL
   según la dirección de mayor probabilidad del ensemble. Único filtro que
   sobrevive: spread instantáneo > `MAX_SPREAD_USD`.

4. **Kelly**: `fraction_to_lots`/`verify_margin` usan `LEVERAGE_XAUUSD` fijo
   en vez de la rama PHASE1/PHASE2.

5. **Backtester/training_loop**: se ajustan para simular el mismo ciclo
   (abre en la vela siguiente a la señal, cierra en el cierre de esa misma
   vela o antes por SL/TP) para que el gate de 59% WR mida la estrategia
   real.

## Archivos afectados

`config/settings.py`, `config/constants.py`, `core/exit_manager.py`,
`core/signal_generator.py`, `core/live_trader.py`, `core/kelly.py`,
`core/backtester.py`, `core/training_loop.py`, `main.py`,
`core/capital_scaler.py` (se deja de invocar), tests relacionados en
`tests/`.

## Riesgo conocido (no bloqueante, documentado por instrucción del usuario)

Con capital $20 y leverage 1:1, el margen no alcanza para el lote mínimo
estándar de XAUUSD. El bot puede entrenar/backtestear normalmente; en vivo,
`kelly.verify_margin` rechazará aperturas hasta que haya capital/leverage
suficiente. No se agrega manejo especial — es el comportamiento ya existente
del sistema de verificación de margen.
