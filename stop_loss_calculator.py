"""
Stop loss dinamico basado en ATR — logica documentada paso a paso porque es
prioridad critica #1 del sistema (debe ser matematicamente preciso y ejecutable
en mercado real, respetando spread y distancia minima del broker).

Calculo:
1. distancia_base = ATR(periodo) * ATR_SL_MULTIPLIER
   -> el SL se mueve con la volatilidad reciente: en mercado tranquilo el stop
      es mas ajustado, en mercado volatil se amplia para no saltar por ruido.
2. distancia_minima_broker = symbol.trade_stops_level * point + MIN_SL_BUFFER_POINTS * point
   -> todo broker exige una distancia minima entre precio y SL/TP (stops_level).
      Operar mas cerca que eso hace que la orden sea rechazada; se aplica ademas
      un colchon extra configurable para absorber slippage de ejecucion.
3. distancia_final = max(distancia_base, distancia_minima_broker)
4. sl_price = entry -+ distancia_final (resta en BUY, suma en SELL)
5. Validacion de spread: si spread_actual > MAX_SPREAD_POINTS, el plan se marca
   invalido (no se opera) porque el spread ya consume una porcion excesiva del
   stop, distorsionando el riesgo real vs el calculado.
6. Take profits (TP1/TP2) se derivan tambien de multiplos de ATR, y se valida
   que el ratio riesgo/beneficio de TP2 contra el SL cumpla MIN_RR_RATIO; si no,
   el plan se invalida (mejor no tomar el trade que tomarlo con RR pobre).

Todos los precios devueltos ya estan redondeados a los `digits` del simbolo.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import config
from indicators import atr

logger = logging.getLogger(__name__)


@dataclass
class StopLossPlan:
    valid: bool
    reason: str
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    sl_distance_price: float = 0.0
    atr_value: float = 0.0
    risk_reward_ratio: float = 0.0


def _round(price: float, digits: int) -> float:
    return round(price, digits)


def calculate_stop_loss_plan(direction: int, entry_price: float, df, symbol_info, spread_points: float) -> StopLossPlan:
    if len(df) < config.ATR_PERIOD + 1:
        return StopLossPlan(valid=False, reason="datos insuficientes para ATR")

    atr_series = atr(df, config.ATR_PERIOD)
    atr_value = atr_series.iloc[-1]

    if atr_value <= 0 or atr_value != atr_value:  # NaN check sin depender de pandas/numpy aqui
        return StopLossPlan(valid=False, reason="ATR invalido o insuficiente")

    if spread_points > config.MAX_SPREAD_POINTS:
        return StopLossPlan(valid=False, reason=f"spread excesivo ({spread_points} > {config.MAX_SPREAD_POINTS})")

    point = symbol_info.point
    digits = symbol_info.digits
    stops_level_price = symbol_info.trade_stops_level * point
    min_buffer_price = config.MIN_SL_BUFFER_POINTS * point

    base_distance = atr_value * config.ATR_SL_MULTIPLIER
    min_distance = stops_level_price + min_buffer_price
    sl_distance = max(base_distance, min_distance)

    tp1_distance = atr_value * config.ATR_TP1_MULTIPLIER
    tp2_distance = atr_value * config.ATR_TP2_MULTIPLIER
    # Los TP tambien deben respetar la distancia minima del broker.
    tp1_distance = max(tp1_distance, min_distance)
    tp2_distance = max(tp2_distance, min_distance)

    if direction == 1:  # BUY
        sl_price = entry_price - sl_distance
        tp1_price = entry_price + tp1_distance
        tp2_price = entry_price + tp2_distance
    elif direction == -1:  # SELL
        sl_price = entry_price + sl_distance
        tp1_price = entry_price - tp1_distance
        tp2_price = entry_price - tp2_distance
    else:
        return StopLossPlan(valid=False, reason="direccion invalida")

    rr_ratio = tp2_distance / sl_distance if sl_distance > 0 else 0.0
    if rr_ratio < config.MIN_RR_RATIO:
        return StopLossPlan(
            valid=False, reason=f"RR insuficiente ({rr_ratio:.2f} < {config.MIN_RR_RATIO})",
            atr_value=atr_value, risk_reward_ratio=rr_ratio,
        )

    return StopLossPlan(
        valid=True, reason="ok",
        entry_price=_round(entry_price, digits),
        sl_price=_round(sl_price, digits),
        tp1_price=_round(tp1_price, digits),
        tp2_price=_round(tp2_price, digits),
        sl_distance_price=sl_distance,
        atr_value=atr_value,
        risk_reward_ratio=rr_ratio,
    )
