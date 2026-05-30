from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("exit_manager")


@dataclass
class ExitAction:
    action:    str           # "HOLD", "CLOSE_PARTIAL", "CLOSE_FULL", "MOVE_SL"
    reason:    str
    new_sl:    Optional[float] = None
    lots:      Optional[float] = None


@dataclass
class OpenPosition:
    ticket:     int
    symbol:     str
    direction:  str
    lots:       float
    lots_remaining: float
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    phase:      int           # 1 = entry→tp1, 2 = trailing activo
    trailing_sl: Optional[float]
    open_time:  datetime
    bars_open:  int = 0


class ExitManager:

    def evaluate(self, position: OpenPosition, bar: dict, atr: float) -> ExitAction:
        """
        Evalúa condición de salida en cada nueva vela cerrada.
        bar: {"open", "high", "low", "close", "time"}
        atr: ATR actual calculado por feature_engine
        """
        position.bars_open += 1

        high  = bar["high"]
        low   = bar["low"]
        close = bar["close"]
        is_buy = position.direction == constants.SIGNAL_BUY

        # ── SL hit ────────────────────────────────────────────────────────────
        if is_buy and low <= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)
        if not is_buy and high >= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)

        # ── Fase 1: TP1 hit ───────────────────────────────────────────────────
        if position.phase == 1:
            tp1_hit = (is_buy and high >= position.tp1) or \
                      (not is_buy and low <= position.tp1)
            if tp1_hit:
                half_lots = round(position.lots * settings.PARTIAL_TP_FRACTION, 2)
                half_lots = max(half_lots, settings.KELLY_MIN_LOTS)
                # Calcula break-even SL
                spread_adj = atr * 0.05
                new_sl = position.entry + spread_adj if is_buy else position.entry - spread_adj
                return ExitAction("CLOSE_PARTIAL", constants.EXIT_TP1,
                                  new_sl=new_sl, lots=half_lots)

        # ── Fase 2: trailing stop ─────────────────────────────────────────────
        if position.phase == 2:
            trail_dist = atr * settings.ATR_TRAILING_MULTIPLIER

            if is_buy:
                new_trail = close - trail_dist
                if position.trailing_sl is None or new_trail > position.trailing_sl:
                    if new_trail > position.sl:
                        return ExitAction("MOVE_SL", "TRAILING", new_sl=new_trail)
                # TP2 hit
                if high >= position.tp2:
                    return ExitAction("CLOSE_FULL", constants.EXIT_TP2)
                # Trailing SL hit
                if position.trailing_sl and low <= position.trailing_sl:
                    return ExitAction("CLOSE_FULL", constants.EXIT_TRAILING)
            else:
                new_trail = close + trail_dist
                if position.trailing_sl is None or new_trail < position.trailing_sl:
                    if new_trail < position.sl:
                        return ExitAction("MOVE_SL", "TRAILING", new_sl=new_trail)
                if low <= position.tp2:
                    return ExitAction("CLOSE_FULL", constants.EXIT_TP2)
                if position.trailing_sl and high >= position.trailing_sl:
                    return ExitAction("CLOSE_FULL", constants.EXIT_TRAILING)

        # ── Time exit: 30 velas sin progreso ──────────────────────────────────
        if position.bars_open >= 30:
            min_progress = atr * 0.2
            progress = (close - position.entry) if is_buy else (position.entry - close)
            if progress < min_progress:
                return ExitAction("CLOSE_FULL", constants.EXIT_TIME)

        return ExitAction("HOLD", "")

    def calc_levels(self, symbol: str, direction: str, entry: float, atr: float) -> dict:
        """Calcula SL, TP1, TP2 para una nueva entrada."""
        sl_dist  = atr * settings.ATR_SL_MULTIPLIER
        tp1_dist = atr * settings.ATR_TP1_MULTIPLIER
        tp2_dist = atr * settings.ATR_TP2_MULTIPLIER

        if direction == constants.SIGNAL_BUY:
            return {
                "sl":  round(entry - sl_dist,  5),
                "tp1": round(entry + tp1_dist, 5),
                "tp2": round(entry + tp2_dist, 5),
            }
        else:
            return {
                "sl":  round(entry + sl_dist,  5),
                "tp1": round(entry - tp1_dist, 5),
                "tp2": round(entry - tp2_dist, 5),
            }
