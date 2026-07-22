"""
Gestion automatica de ganancias: cierre en el punto de maxima ganancia
detectado, trailing stop dinamico y take-profit escalonado.

Por cada posicion abierta se mantiene un `PositionTracker` que registra el
profit flotante maximo alcanzado ("peak"). El trailing stop solo se arma
despues de superar un profit minimo (para no cerrar en breakeven+ruido), y una
vez armado nunca retrocede: solo se mueve a favor. Si el profit flotante cae
mas de `PROFIT_GIVEBACK_TOLERANCE_PCT` respecto al peak, se cierra la posicion
completa en mercado — esa es la implementacion concreta de "cerrar en el punto
de maxima ganancia" (no se puede saber el pico exacto hasta que ya paso, asi
que se cierra apenas se confirma la reversion, dentro de la tolerancia
configurada).

Take-profit escalonado (TP1/TP2) cierra fracciones de volumen en el camino,
independientemente del trailing, para asegurar ganancia parcial temprano.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class ActionType(Enum):
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    MODIFY_SL = "MODIFY_SL"
    FULL_CLOSE = "FULL_CLOSE"


@dataclass
class ProfitAction:
    type: ActionType
    volume: Optional[float] = None
    sl_price: Optional[float] = None
    reason: str = ""


@dataclass
class PositionTracker:
    entry_price: float
    direction: int
    initial_volume: float
    tp1_price: float
    tp2_price: float
    tp1_done: bool = False
    tp2_done: bool = False
    trailing_armed: bool = False
    peak_profit_price_units: float = 0.0
    last_trailing_sl: Optional[float] = None


class ProfitManager:
    def __init__(self):
        self._trackers: Dict[int, PositionTracker] = {}

    def register_position(self, ticket: int, entry_price: float, direction: int,
                           volume: float, tp1_price: float, tp2_price: float):
        self._trackers[ticket] = PositionTracker(
            entry_price=entry_price, direction=direction, initial_volume=volume,
            tp1_price=tp1_price, tp2_price=tp2_price,
        )

    def forget_position(self, ticket: int):
        self._trackers.pop(ticket, None)

    def has_position(self, ticket: int) -> bool:
        return ticket in self._trackers

    def evaluate(self, ticket: int, current_price: float, current_sl: float,
                 atr_value: float, min_stop_distance: float = 0.0) -> List[ProfitAction]:
        tracker = self._trackers.get(ticket)
        if tracker is None:
            return []

        direction = tracker.direction
        floating_profit = (current_price - tracker.entry_price) * direction
        actions: List[ProfitAction] = []

        if floating_profit <= 0:
            return actions  # nunca se gestiona "ganancia maxima" mientras el trade esta en perdida

        # --- Take profit escalonado -----------------------------------
        tp1_distance = abs(tracker.tp1_price - tracker.entry_price)
        tp2_distance = abs(tracker.tp2_price - tracker.entry_price)

        if not tracker.tp1_done and floating_profit >= tp1_distance:
            vol = round(tracker.initial_volume * config.TP1_CLOSE_FRACTION, 2)
            if vol > 0:
                actions.append(ProfitAction(ActionType.PARTIAL_CLOSE, volume=vol, reason="TP1 alcanzado"))
            tracker.tp1_done = True

        if not tracker.tp2_done and floating_profit >= tp2_distance:
            vol = round(tracker.initial_volume * config.TP2_CLOSE_FRACTION, 2)
            if vol > 0:
                actions.append(ProfitAction(ActionType.PARTIAL_CLOSE, volume=vol, reason="TP2 alcanzado"))
            tracker.tp2_done = True

        # --- Trailing / cierre en maxima ganancia -----------------------
        tracker.peak_profit_price_units = max(tracker.peak_profit_price_units, floating_profit)

        arm_threshold = config.ATR_TRAIL_ARM_MULTIPLIER * atr_value
        if not tracker.trailing_armed and floating_profit >= arm_threshold:
            tracker.trailing_armed = True
            logger.info("Trailing armado para ticket %d (profit=%.5f)", ticket, floating_profit)

        if tracker.trailing_armed:
            giveback = tracker.peak_profit_price_units - floating_profit
            giveback_tolerance = tracker.peak_profit_price_units * (config.PROFIT_GIVEBACK_TOLERANCE_PCT / 100.0)

            if giveback > giveback_tolerance:
                actions.append(ProfitAction(
                    ActionType.FULL_CLOSE,
                    reason=f"reversion desde maxima ganancia (peak={tracker.peak_profit_price_units:.5f}, "
                           f"actual={floating_profit:.5f})",
                ))
                self.forget_position(ticket)
                return actions

            # El trailing nunca puede quedar mas cerca del precio que la
            # distancia minima del broker (stops_level), o el modify seria rechazado.
            trail_distance = max(config.ATR_TRAIL_DISTANCE_MULTIPLIER * atr_value, min_stop_distance)
            candidate_sl = current_price - direction * trail_distance
            improves = (
                tracker.last_trailing_sl is None
                or (direction == 1 and candidate_sl > tracker.last_trailing_sl)
                or (direction == -1 and candidate_sl < tracker.last_trailing_sl)
            )
            also_better_than_current = (
                (direction == 1 and candidate_sl > current_sl)
                or (direction == -1 and candidate_sl < current_sl)
            )
            if improves and also_better_than_current:
                tracker.last_trailing_sl = candidate_sl
                actions.append(ProfitAction(ActionType.MODIFY_SL, sl_price=candidate_sl, reason="trailing stop"))

        return actions
