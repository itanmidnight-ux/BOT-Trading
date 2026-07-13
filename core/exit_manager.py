from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("exit_manager")


@dataclass
class ExitAction:
    action:    str           # siempre "CLOSE_FULL" en el ciclo de 1 vela
    reason:    str


@dataclass
class OpenPosition:
    ticket:     int
    symbol:     str
    direction:  str
    lots:       float
    entry:      float
    sl:         float
    tp:         float
    open_time:  datetime
    bars_open:  int = 0


class ExitManager:
    """
    Ciclo de 1 vela: cada posición se abre y se cierra dentro del mismo
    bar M1. evaluate() siempre devuelve CLOSE_FULL — por SL, por TP, o
    forzado al cierre de la vela si no se tocó ninguno de los dos.
    """

    def evaluate(self, position: OpenPosition, bar: dict, atr: float) -> ExitAction:
        position.bars_open += 1

        high, low, close = bar["high"], bar["low"], bar["close"]
        is_buy = position.direction == constants.SIGNAL_BUY

        if is_buy and low <= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)
        if not is_buy and high >= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)

        tp_hit = (is_buy and high >= position.tp) or (not is_buy and low <= position.tp)
        if tp_hit:
            return ExitAction("CLOSE_FULL", constants.EXIT_TP)

        return ExitAction("CLOSE_FULL", constants.EXIT_BAR_CLOSE)

    def calc_levels(self, symbol: str, direction: str, entry: float, atr: float) -> dict:
        """Calcula SL y TP para una nueva entrada (hold de 1 vela)."""
        sl_dist = atr * settings.ATR_SL_MULTIPLIER
        tp_dist = atr * settings.ATR_TP1_MULTIPLIER

        if direction == constants.SIGNAL_BUY:
            return {
                "sl": round(entry - sl_dist, 2),
                "tp": round(entry + tp_dist, 2),
            }
        return {
            "sl": round(entry + sl_dist, 2),
            "tp": round(entry - tp_dist, 2),
        }
