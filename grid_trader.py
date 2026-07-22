"""
Grid trading como red de contencion de perdidas (no como estrategia de entrada
por si misma). Cuando una posicion base se mueve en contra, el grid escala
posiciones adicionales en niveles predefinidos por ATR para mejorar el precio
promedio, PERO con un tope duro de riesgo agregado (GRID_MAX_TOTAL_RISK_PCT del
equity): si escalar el siguiente nivel superaria ese tope, el nivel se descarta
y el "basket_stop_price" (un nivel mas alla del ultimo nivel incluido) pasa a
ser el corte de perdida absoluto de toda la canasta.

Esto es deliberadamente conservador respecto a un grid/martingala clasico sin
limite: aqui el numero de niveles y el riesgo total estan acotados de antemano,
nunca se escala "a ciegas" mas alla del presupuesto de riesgo configurado.

Advertencia de diseño: promediar posiciones perdedoras aumenta la exposicion
antes de que se confirme que el mercado se dio vuelta. El tope de riesgo
agregado existe justamente para que, en el peor caso, la perdida maxima de la
canasta completa siga estando acotada y sea conocida de antemano.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    index: int
    trigger_price: float
    volume: float
    opened: bool = False
    ticket: Optional[int] = None


@dataclass
class GridSession:
    direction: int
    base_entry_price: float
    atr_value: float
    step_price: float
    levels: List[GridLevel]
    basket_stop_price: float
    base_ticket: Optional[int] = None

    def total_volume_opened(self) -> float:
        return sum(l.volume for l in self.levels if l.opened)

    def opened_level_tickets(self) -> List[int]:
        return [l.ticket for l in self.levels if l.opened and l.ticket is not None and l.ticket > 0]

    def next_pending_level(self, current_price: float) -> Optional[GridLevel]:
        for level in self.levels:
            if level.opened:
                continue
            if self.direction == 1 and current_price <= level.trigger_price:
                return level
            if self.direction == -1 and current_price >= level.trigger_price:
                return level
        return None

    def mark_opened(self, index: int, ticket: int):
        for level in self.levels:
            if level.index == index:
                level.opened = True
                level.ticket = ticket
                return

    def basket_stop_hit(self, current_price: float) -> bool:
        if self.direction == 1:
            return current_price <= self.basket_stop_price
        return current_price >= self.basket_stop_price


def _value_per_price_unit_per_lot(symbol_info) -> float:
    if symbol_info.trade_tick_size <= 0:
        return 0.0
    return symbol_info.trade_tick_value / symbol_info.trade_tick_size


def build_grid_session(direction: int, entry_price: float, atr_value: float, base_volume: float,
                        sl_distance_price: float, symbol_info, equity: float) -> GridSession:
    if not config.GRID_ENABLED or atr_value <= 0:
        return GridSession(direction, entry_price, atr_value, 0.0, [], entry_price)

    step = config.GRID_STEP_ATR_MULTIPLIER * atr_value
    step_size_symbol = symbol_info.volume_step or config.MIN_LOT
    value_per_unit = _value_per_price_unit_per_lot(symbol_info)

    max_total_risk = equity * (config.GRID_MAX_TOTAL_RISK_PCT / 100.0)
    base_risk = base_volume * sl_distance_price * value_per_unit
    cumulative_risk = base_risk

    levels: List[GridLevel] = []
    included_count = 0

    for i in range(1, config.GRID_LEVELS + 1):
        raw_volume = base_volume * (config.GRID_LOT_MULTIPLIER ** i)
        volume = math.floor(raw_volume / step_size_symbol) * step_size_symbol
        volume = min(volume, symbol_info.volume_max)
        if volume < symbol_info.volume_min:
            break

        level_sl_distance = sl_distance_price + i * step  # el stop efectivo se aleja con cada nivel de precio promedio
        level_risk = volume * level_sl_distance * value_per_unit

        if cumulative_risk + level_risk > max_total_risk:
            logger.info("Grid detenido en nivel %d: riesgo acumulado excederia el tope (%.2f > %.2f)",
                        i, cumulative_risk + level_risk, max_total_risk)
            break

        cumulative_risk += level_risk
        trigger_price = entry_price - i * step if direction == 1 else entry_price + i * step
        levels.append(GridLevel(index=i, trigger_price=trigger_price, volume=volume))
        included_count = i

    basket_stop_distance = (included_count + 1) * step if included_count > 0 else sl_distance_price
    basket_stop_price = (entry_price - basket_stop_distance if direction == 1
                          else entry_price + basket_stop_distance)

    return GridSession(
        direction=direction, base_entry_price=entry_price, atr_value=atr_value,
        step_price=step, levels=levels, basket_stop_price=basket_stop_price,
    )
