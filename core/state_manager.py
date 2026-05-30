import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from config import settings

_log = get_logger("state_manager")
_STATE_FILE = settings.LOGS_SYSTEM_DIR / "state.json"


class StateManager:
    def __init__(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict = self._load_raw()

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _load_raw(self) -> dict:
        if _STATE_FILE.exists():
            try:
                return json.loads(_STATE_FILE.read_text())
            except Exception as e:
                _log.warning(f"Estado corrupto, reiniciando: {e}")
        return {"positions": {}, "capital": settings.INITIAL_CAPITAL,
                "trades_today": 0, "consecutive_losses": 0, "phase": 1}

    def _save(self):
        tmp = _STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, default=str))
        tmp.replace(_STATE_FILE)

    # ── Positions ─────────────────────────────────────────────────────────────

    def save_position(self, symbol: str, ticket: int, direction: str,
                      lots: float, entry: float, sl: float, tp1: float, tp2: float):
        self._state["positions"][symbol] = {
            "ticket":          ticket,
            "direction":       direction,
            "lots":            lots,
            "lots_remaining":  lots,
            "entry":           entry,
            "sl":              sl,
            "tp1":             tp1,
            "tp2":             tp2,
            "phase":           1,
            "trailing_sl":     None,
            "bars_open":       0,
            "open_time":       datetime.utcnow().isoformat(),
        }
        self._save()
        _log.debug(f"Estado guardado: {symbol} ticket:{ticket}")

    def update_position(self, symbol: str, **kwargs):
        if symbol in self._state["positions"]:
            self._state["positions"][symbol].update(kwargs)
            self._save()

    def clear_position(self, symbol: str):
        if symbol in self._state["positions"]:
            del self._state["positions"][symbol]
            self._save()

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self._state["positions"]

    def get_position(self, symbol: str) -> Optional[dict]:
        return self._state["positions"].get(symbol)

    def get_all_positions(self) -> dict:
        return self._state["positions"]

    # ── Capital & stats ───────────────────────────────────────────────────────

    @property
    def capital(self) -> float:
        return self._state.get("capital", settings.INITIAL_CAPITAL)

    @capital.setter
    def capital(self, value: float):
        self._state["capital"] = round(value, 4)
        self._save()

    @property
    def phase(self) -> int:
        return self._state.get("phase", 1)

    @phase.setter
    def phase(self, value: int):
        self._state["phase"] = value
        self._save()

    @property
    def consecutive_losses(self) -> int:
        return self._state.get("consecutive_losses", 0)

    def record_trade_result(self, win: bool):
        today = datetime.utcnow().strftime("%Y%m%d")
        if self._state.get("trades_date") != today:
            self._state["trades_today"] = 0
            self._state["trades_date"]  = today
        self._state["trades_today"] = self._state.get("trades_today", 0) + 1
        if win:
            self._state["consecutive_losses"] = 0
        else:
            self._state["consecutive_losses"] = self._state.get("consecutive_losses", 0) + 1
        self._save()

    # ── Verificación contra MT5 al arrancar ──────────────────────────────────

    def verify_with_mt5(self, trade_manager) -> list:
        """
        Compara state con posiciones reales en MT5.
        Retorna lista de acciones tomadas.
        """
        actions = []
        mt5_positions = {str(p.ticket): p for p in trade_manager.get_open_positions()}

        for symbol, pos_data in list(self._state["positions"].items()):
            ticket = str(pos_data.get("ticket", ""))
            if ticket not in mt5_positions:
                _log.warning(f"Posición {symbol} ticket:{ticket} no existe en MT5 — limpiando estado")
                self.clear_position(symbol)
                actions.append(f"CLEANED_{symbol}")

        for ticket_str, mt5_pos in mt5_positions.items():
            symbol = mt5_pos.symbol
            if not self.has_open_position(symbol):
                _log.warning(f"MT5 tiene posición {symbol} ticket:{ticket_str} sin estado — adoptando")
                from config.constants import SIGNAL_BUY, SIGNAL_SELL
                direction = SIGNAL_BUY if mt5_pos.type == 0 else SIGNAL_SELL
                self.save_position(symbol, mt5_pos.ticket, direction,
                                   mt5_pos.volume, mt5_pos.price_open,
                                   mt5_pos.sl, mt5_pos.tp, mt5_pos.tp)
                actions.append(f"ADOPTED_{symbol}")

        return actions
