"""
Wrapper directo sobre el paquete `MetaTrader5` (API oficial de Python).
Este es el camino PRIMARIO de ejecucion: mas simple y robusto que pasar por el
EA para cuentas donde el broker permite trading algoritmico via API. El EA
(`expert_advisor.mq5` + `mt5_connector/bridge.py`) es la via alternativa quando
se requiere que un EA dentro del terminal sea quien coloque las ordenes.

Requiere que el terminal MT5 este corriendo y logueado (o credenciales validas
en config.py) antes de llamar a `connect()`.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config
from mt5_compat import mt5  # ver mt5_compat.py: resuelve MetaTrader5 nativo o via bridge mt5linux

logger = logging.getLogger(__name__)

_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


@dataclass
class AccountSnapshot:
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    currency: str
    leverage: int


class MT5ConnectionError(RuntimeError):
    pass


class MT5Connector:
    def __init__(self):
        self._connected = False

    # ------------------------------------------------------------------
    # Conexion / reconexion
    # ------------------------------------------------------------------
    def connect(self) -> None:
        kwargs = {}
        if config.MT5_TERMINAL_PATH:
            kwargs["path"] = config.MT5_TERMINAL_PATH
        if config.MT5_LOGIN and config.MT5_PASSWORD and config.MT5_SERVER:
            kwargs.update(login=int(config.MT5_LOGIN), password=config.MT5_PASSWORD, server=config.MT5_SERVER)

        if not mt5.initialize(**kwargs):
            raise MT5ConnectionError(f"mt5.initialize() fallo: {mt5.last_error()}")

        if not mt5.symbol_select(config.SYMBOL, True):
            raise MT5ConnectionError(f"No se pudo seleccionar el simbolo {config.SYMBOL}: {mt5.last_error()}")

        self._connected = True
        logger.info("Conectado a MT5 (%s)", config.SYMBOL)

    def ensure_connected(self) -> None:
        if self._connected and mt5.terminal_info() is not None:
            return
        logger.warning("Conexion MT5 perdida, reintentando...")
        last_err = None
        for attempt in range(1, config.MT5_RECONNECT_ATTEMPTS + 1):
            try:
                self.connect()
                return
            except MT5ConnectionError as e:
                last_err = e
                wait = config.MT5_RECONNECT_BACKOFF_SEC * attempt
                logger.warning("Reintento %d/%d fallo (%s), esperando %.1fs", attempt,
                                config.MT5_RECONNECT_ATTEMPTS, e, wait)
                time.sleep(wait)
        raise MT5ConnectionError(f"No se pudo reconectar tras {config.MT5_RECONNECT_ATTEMPTS} intentos: {last_err}")

    def shutdown(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    # ------------------------------------------------------------------
    # Cuenta / capital
    # ------------------------------------------------------------------
    def get_account_snapshot(self) -> AccountSnapshot:
        self.ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError(f"account_info() devolvio None: {mt5.last_error()}")
        return AccountSnapshot(
            balance=info.balance, equity=info.equity, margin=info.margin,
            margin_free=info.margin_free, margin_level=info.margin_level or 0.0,
            currency=info.currency, leverage=info.leverage,
        )

    # ------------------------------------------------------------------
    # Datos de mercado
    # ------------------------------------------------------------------
    def get_symbol_info(self):
        self.ensure_connected()
        info = mt5.symbol_info(config.SYMBOL)
        if info is None:
            raise MT5ConnectionError(f"symbol_info({config.SYMBOL}) devolvio None: {mt5.last_error()}")
        return info

    def get_rates(self, timeframe: str = None, count: int = None) -> pd.DataFrame:
        self.ensure_connected()
        tf_name = _TIMEFRAME_MAP[timeframe or config.TIMEFRAME]
        tf_const = getattr(mt5, tf_name)
        count = count or config.BARS_LOOKBACK
        rates = mt5.copy_rates_from_pos(config.SYMBOL, tf_const, 0, count)
        if rates is None or len(rates) == 0:
            raise MT5ConnectionError(f"copy_rates_from_pos devolvio vacio: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_tick(self):
        self.ensure_connected()
        tick = mt5.symbol_info_tick(config.SYMBOL)
        if tick is None:
            raise MT5ConnectionError(f"symbol_info_tick devolvio None: {mt5.last_error()}")
        return tick

    def get_spread_points(self) -> float:
        info = self.get_symbol_info()
        return info.spread

    # ------------------------------------------------------------------
    # Posiciones
    # ------------------------------------------------------------------
    def get_open_positions(self) -> List:
        self.ensure_connected()
        positions = mt5.positions_get(symbol=config.SYMBOL)
        return list(positions) if positions else []

    def get_position_realized_pnl(self, ticket: int) -> float:
        """Suma profit + comision + swap de todos los deals asociados a una
        posicion ya cerrada, consultando el historial de la cuenta."""
        self.ensure_connected()
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return 0.0
        return sum(d.profit + d.commission + d.swap for d in deals)

    # ------------------------------------------------------------------
    # Ordenes
    # ------------------------------------------------------------------
    def _order_filling_mode(self):
        """Elige el filling mode que el simbolo realmente soporta (bitmask
        SYMBOL_FILLING_*). Muchos brokers de XAUUSD solo aceptan FOK; con IOC
        hardcodeado esas cuentas rechazarian todas las ordenes."""
        info = self.get_symbol_info()
        modes = getattr(info, "filling_mode", 0)
        if modes & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        if modes & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def send_market_order(self, direction: int, volume: float, sl: float, tp: Optional[float],
                           comment: str = "", magic: int = None):
        """direction: 1 = BUY, -1 = SELL. Devuelve el OrderSendResult de mt5."""
        self.ensure_connected()
        tick = self.get_tick()
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": config.SYMBOL,
            "volume": round(volume, 2),
            "type": order_type,
            "price": price,
            "sl": sl,
            "deviation": config.SLIPPAGE_POINTS,
            "magic": magic or config.BRIDGE_MAGIC,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._order_filling_mode(),
        }
        if tp is not None:
            request["tp"] = tp

        if config.DRY_RUN:
            logger.info("[DRY_RUN] order_send omitido: %s", request)
            return None

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("order_send fallo: %s | request=%s", result, request)
        return result

    def close_position(self, position, volume: Optional[float] = None, comment: str = ""):
        self.ensure_connected()
        tick = self.get_tick()
        close_volume = volume if volume is not None else position.volume
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": config.SYMBOL,
            "volume": round(close_volume, 2),
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": config.SLIPPAGE_POINTS,
            "magic": position.magic,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._order_filling_mode(),
        }

        if config.DRY_RUN:
            logger.info("[DRY_RUN] close omitido: %s", request)
            return None

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("close_position fallo: %s | request=%s", result, request)
        return result

    def modify_sltp(self, position, sl: Optional[float] = None, tp: Optional[float] = None):
        self.ensure_connected()
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": config.SYMBOL,
            "position": position.ticket,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
            "magic": position.magic,
        }

        if config.DRY_RUN:
            logger.info("[DRY_RUN] modify_sltp omitido: %s", request)
            return None

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("modify_sltp fallo: %s | request=%s", result, request)
        return result
