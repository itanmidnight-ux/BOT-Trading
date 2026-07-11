import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("trade_manager")

BOT_MAGIC = 20260530


@dataclass
class TradeResult:
    ticket:     int
    symbol:     str
    direction:  str
    lots:       float
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    open_time:  datetime
    retcode:    int
    success:    bool
    message:    str = ""


@dataclass
class ClosedTrade:
    ticket:     int
    symbol:     str
    direction:  str
    lots:       float
    entry:      float
    exit_price: float
    sl:         float
    tp1:        float
    pips:       float
    pnl_usd:    float
    reason:     str
    open_time:  datetime
    close_time: datetime = field(default_factory=datetime.utcnow)


class TradeManager:
    def __init__(self):
        self._trades_dir = settings.LOGS_TRADES_DIR
        self._trades_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def open_market_order(self, symbol: str, direction: str, lots: float,
                          sl_price: float, tp_price: float) -> TradeResult:
        try:
            from core.mt5_compat import mt5
        except Exception:
            return TradeResult(0, symbol, direction, lots, 0, sl_price, tp_price,
                               0, datetime.utcnow(), -1, False, "MT5 no disponible")

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return TradeResult(0, symbol, direction, lots, 0, sl_price, tp_price,
                               0, datetime.utcnow(), -1, False, f"No tick para {symbol}")

        order_type = mt5.ORDER_TYPE_BUY if direction == constants.SIGNAL_BUY else mt5.ORDER_TYPE_SELL
        price      = tick.ask if direction == constants.SIGNAL_BUY else tick.bid

        # TP2 se gestiona via exit_manager con trailing; solo ponemos TP1 en MT5
        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      lots,
            "type":        order_type,
            "price":       price,
            "sl":          round(sl_price, mt5.symbol_info(symbol).digits),
            "tp":          round(tp_price, mt5.symbol_info(symbol).digits),
            "deviation":   30,
            "magic":       BOT_MAGIC,
            "comment":     "BOT-Trading",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = self._send_with_retry(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE if result else False

        ticket  = result.order if success else 0
        entry   = result.price if success else price
        message = self._retcode_msg(result.retcode) if result else "Sin respuesta"

        trade = TradeResult(ticket, symbol, direction, lots, entry, sl_price,
                            tp_price, 0.0, datetime.utcnow(), result.retcode if result else -1,
                            success, message)
        if success:
            _log.info(f"OPEN {direction} {symbol} {lots}lots @ {entry:.5f} SL:{sl_price:.5f} ticket:{ticket}")
        else:
            _log.error(f"OPEN FAILED {symbol}: {message}")

        return trade

    def close_position(self, ticket: int, symbol: str, direction: str,
                       lots: Optional[float] = None) -> bool:
        try:
            from core.mt5_compat import mt5
        except Exception:
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            _log.warning(f"Posición {ticket} no encontrada")
            return False

        pos = positions[0]
        close_lots  = lots if lots else pos.volume
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick        = mt5.symbol_info_tick(symbol)
        close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      close_lots,
            "type":        close_type,
            "position":    ticket,
            "price":       close_price,
            "deviation":   30,
            "magic":       BOT_MAGIC,
            "comment":     "BOT-Close",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = self._send_with_retry(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE if result else False

        if success:
            pips = self._calc_pips(symbol, pos.price_open, close_price, direction)
            pnl  = pos.profit if hasattr(pos, 'profit') else 0.0
            self._log_closed_trade(ticket, symbol, direction, close_lots,
                                   pos.price_open, close_price, pips, pnl, "MANUAL")
            _log.info(f"CLOSE {symbol} ticket:{ticket} @ {close_price:.5f} pnl:{pnl:+.2f}")
        else:
            _log.error(f"CLOSE FAILED ticket:{ticket}: {self._retcode_msg(result.retcode if result else -1)}")

        return success

    def close_all_positions(self, symbol: Optional[str] = None) -> int:
        try:
            from core.mt5_compat import mt5
        except Exception:
            return 0

        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return 0

        closed = 0
        for pos in positions:
            if pos.magic == BOT_MAGIC:
                direction = constants.SIGNAL_BUY if pos.type == 0 else constants.SIGNAL_SELL
                if self.close_position(pos.ticket, pos.symbol, direction):
                    closed += 1
        return closed

    def modify_position_sl(self, ticket: int, symbol: str, new_sl: float) -> bool:
        try:
            from core.mt5_compat import mt5
        except Exception:
            return False

        info = mt5.symbol_info(symbol)
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   symbol,
            "sl":       round(new_sl, info.digits if info else 5),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            _log.debug(f"SL modificado ticket:{ticket} nuevo SL:{new_sl:.5f}")
            return True
        _log.warning(f"Fallo modificar SL ticket:{ticket}")
        return False

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        try:
            from core.mt5_compat import mt5
        except Exception:
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return []
        return [p for p in positions if p.magic == BOT_MAGIC]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _send_with_retry(self, request, max_retries: int = 3):
        from core.mt5_compat import mt5
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            if result is None:
                _log.warning(f"order_send retornó None (intento {attempt+1})")
                time.sleep(2 ** attempt)
                continue
            if result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF):
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick:
                    is_buy = request["type"] in (mt5.ORDER_TYPE_BUY,)
                    request["price"] = tick.ask if is_buy else tick.bid
                time.sleep(0.5)
                continue
            return result
        return None

    @staticmethod
    def _retcode_msg(retcode: int) -> str:
        msgs = {
            0:  "OK",
            10004: "REQUOTE",
            10006: "REQUEST_REJECTED",
            10007: "REQUEST_CANCEL",
            10009: "REQUEST_DONE",
            10013: "INVALID_PARAMS",
            10014: "INVALID_VOLUME",
            10015: "INVALID_PRICE",
            10016: "INVALID_STOPS",
            10017: "TRADE_DISABLED",
            10018: "MARKET_CLOSED",
            10019: "INSUFFICIENT_FUNDS",
            10020: "PRICES_CHANGED",
            10027: "REQUEST_TOO_MANY",
        }
        return msgs.get(retcode, f"CODE_{retcode}")

    @staticmethod
    def _calc_pips(symbol: str, entry: float, exit_price: float, direction: str) -> float:
        diff = exit_price - entry if direction == constants.SIGNAL_BUY else entry - exit_price
        divisor = 0.00010 if "XAU" not in symbol else 0.1
        return round(diff / divisor, 1)

    def _log_closed_trade(self, ticket, symbol, direction, lots,
                          entry, exit_price, pips, pnl, reason):
        row = {
            "timestamp":   datetime.utcnow().isoformat(),
            "ticket":      ticket,
            "symbol":      symbol,
            "direction":   direction,
            "lots":        lots,
            "entry":       entry,
            "exit_price":  exit_price,
            "pips":        pips,
            "pnl_usd":     round(pnl, 4),
            "reason":      reason,
        }
        date_str = datetime.utcnow().strftime("%Y%m%d")
        path     = self._trades_dir / f"trades_{date_str}.csv"
        df_row   = pd.DataFrame([row])
        if path.exists():
            df_row.to_csv(path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(path, index=False)
