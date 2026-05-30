"""
CapitalScaler — Manages phased capital deployment and symbol activation.

Phases:
    Phase 1  equity < $30   → only EURUSD, leverage 3000
    Phase 2  equity >= $30  → EURUSD + XAUUSD, leverage 1000 for XAU

Transitions:
    PHASE_UP    equity crosses $30 upward   → activate XAUUSD
    PHASE_DOWN  equity falls below $25 while in phase 2 → deactivate XAUUSD temporarily

Phase history is persisted to logs/system/phase_history.csv.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import settings
from utils.logger import get_logger
from utils.notifier import notify, PHASE_CHANGE

_log = get_logger("capital_scaler")

_PHASE_HISTORY_COLS = ["timestamp", "event", "phase", "symbol", "equity"]

# Hysteresis band: PHASE_UP at $30, PHASE_DOWN below $25 (from phase 2)
_PHASE_UP_THRESHOLD: float = settings.CAPITAL_PHASE2_THRESHOLD        # 30.0
_PHASE_DOWN_THRESHOLD: float = _PHASE_UP_THRESHOLD * (25 / 30)         # ≈ 25.0


class CapitalScaler:
    """
    Tracks the current trading phase and the set of active instruments.

    Designed to be called once per bar / capital update.
    """

    def __init__(self) -> None:
        self._history_path = Path(settings.LOGS_SYSTEM_DIR) / "phase_history.csv"
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        # Start from a deterministic state; _load_history may override.
        self._current_phase: int = 1
        self._load_history()

    # ── private helpers ────────────────────────────────────────────────────────

    def _load_history(self) -> None:
        """Read the last recorded phase from phase_history.csv."""
        if not self._history_path.exists():
            return
        try:
            last_row: Optional[dict] = None
            with open(self._history_path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    last_row = row
            if last_row:
                self._current_phase = int(last_row.get("phase", 1))
                _log.info(
                    "CapitalScaler: restored phase=%d from history (event=%s equity=%s)",
                    self._current_phase,
                    last_row.get("event"),
                    last_row.get("equity"),
                )
        except Exception as exc:
            _log.warning("CapitalScaler: could not load phase_history.csv — %s", exc)

    def _write_history(
        self,
        event: str,
        phase: int,
        symbol: str,
        equity: float,
    ) -> None:
        write_header = (
            not self._history_path.exists()
            or self._history_path.stat().st_size == 0
        )
        try:
            with open(self._history_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_PHASE_HISTORY_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "event": event,
                    "phase": phase,
                    "symbol": symbol,
                    "equity": round(equity, 2),
                })
        except Exception as exc:
            _log.error("CapitalScaler: failed to write phase_history.csv — %s", exc)

    # ── public API ─────────────────────────────────────────────────────────────

    def get_phase(self, equity: float) -> int:
        """Return the current phase (1 or 2) for *equity* without changing state."""
        if equity >= _PHASE_UP_THRESHOLD:
            return 2
        return 1

    def get_active_symbols(self, equity: float) -> List[str]:
        """
        Return the list of tradeable symbols for the given *equity* level.

        Phase 1: ["EURUSD"]
        Phase 2: ["EURUSD", "XAUUSD"]
        """
        if self.get_phase(equity) == 2:
            return [settings.SYMBOL_PHASE1, settings.SYMBOL_PHASE2]
        return [settings.SYMBOL_PHASE1]

    def get_leverage(self, symbol: str) -> int:
        """Return the configured leverage for *symbol*."""
        if "XAU" in symbol:
            return settings.LEVERAGE_PHASE2   # 1000
        return settings.LEVERAGE_PHASE1       # 3000

    def check_transition(
        self,
        old_equity: float,
        new_equity: float,
    ) -> Optional[dict]:
        """
        Detect a phase transition between two equity snapshots.

        Returns a transition dict on change, None otherwise:
            {"event": "PHASE_UP"|"PHASE_DOWN", "phase": int, "symbol": str}

        State (self._current_phase) is updated here.
        """
        old_phase = self._current_phase
        new_phase = old_phase  # tentative

        # PHASE_UP: equity crosses the activation threshold upward
        if old_phase == 1 and new_equity >= _PHASE_UP_THRESHOLD:
            new_phase = 2

        # PHASE_DOWN: equity drops below the hysteresis threshold while in phase 2
        elif old_phase == 2 and new_equity < _PHASE_DOWN_THRESHOLD:
            new_phase = 1

        if new_phase == old_phase:
            return None

        # Transition detected
        self._current_phase = new_phase

        if new_phase > old_phase:
            event = "PHASE_UP"
            symbol = settings.SYMBOL_PHASE2   # XAUUSD activated
        else:
            event = "PHASE_DOWN"
            symbol = settings.SYMBOL_PHASE2   # XAUUSD deactivated

        transition = {"event": event, "phase": new_phase, "symbol": symbol}
        _log.info(
            "check_transition: %s | phase %d→%d | equity %.2f→%.2f",
            event, old_phase, new_phase, old_equity, new_equity,
        )
        self.log_phase_event({**transition, "equity": new_equity})
        return transition

    def log_phase_event(self, event: dict) -> None:
        """
        Persist a phase event to phase_history.csv and broadcast a notification.

        *event* must contain keys: event, phase, symbol, equity.
        """
        evt = event.get("event", "UNKNOWN")
        phase = event.get("phase", self._current_phase)
        symbol = event.get("symbol", "")
        equity = float(event.get("equity", 0.0))

        self._write_history(evt, phase, symbol, equity)

        notify(
            PHASE_CHANGE,
            {
                "phase": phase,
                "capital": equity,
                "symbol": symbol,
            },
        )
        _log.info("log_phase_event: %s phase=%d symbol=%s equity=%.2f", evt, phase, symbol, equity)
