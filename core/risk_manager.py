"""
RiskManager — Gate-keeps every trade entry with a multi-layer risk check.

Risk checks performed (all must pass):
    1. Daily loss limit
    2. Maximum drawdown
    3. Consecutive losses
    4. Margin level
    5. Volatility (ATR spike)
    6. Trading-hour filter

Emergency stop closes all open positions and writes to risk_events.csv.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from config import settings
from utils.logger import get_logger
from utils.notifier import notify, RISK_EVENT

_log = get_logger("risk_manager")

_RISK_EVENTS_COLS = ["timestamp", "event", "reason", "duration_hours", "equity"]
_TRADE_LOG_COLS = ["timestamp", "symbol", "direction", "lots", "entry", "exit",
                   "pnl", "pips", "reason"]


class RiskManager:
    """
    Centralised risk gate for the trading bot.

    All public check_* methods return (ok: bool, reason: str).
    check_all() aggregates every check and returns on the first failure.
    """

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.equity_peak = initial_capital
        self._risk_events_path = Path(settings.LOGS_SYSTEM_DIR) / "risk_events.csv"
        self._risk_events_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_equity_peak()

    # ── private helpers ────────────────────────────────────────────────────────

    def _load_equity_peak(self) -> None:
        """Scan today's trade log to rebuild equity_peak from persisted data."""
        trades_dir = Path(settings.LOGS_TRADES_DIR)
        if not trades_dir.exists():
            return
        running = self.initial_capital
        peak = self.initial_capital
        # Read all trade files in chronological order
        for trade_file in sorted(trades_dir.glob("trades_*.csv")):
            try:
                with open(trade_file, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        try:
                            running += float(row.get("pnl", 0.0))
                            if running > peak:
                                peak = running
                        except (ValueError, TypeError):
                            continue
            except Exception as exc:
                _log.warning("_load_equity_peak: could not read %s — %s", trade_file, exc)
        self.equity_peak = peak
        _log.info("RiskManager: equity_peak loaded = %.2f", self.equity_peak)

    def _today_trade_path(self) -> Path:
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        return Path(settings.LOGS_TRADES_DIR) / f"trades_{today}.csv"

    def _read_today_trades(self) -> list[dict]:
        path = self._today_trade_path()
        if not path.exists():
            return []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except Exception as exc:
            _log.warning("_read_today_trades: %s", exc)
            return []

    def _write_risk_event(
        self,
        event: str,
        reason: str,
        duration_hours: float,
        equity: float,
    ) -> None:
        write_header = (
            not self._risk_events_path.exists()
            or self._risk_events_path.stat().st_size == 0
        )
        try:
            with open(self._risk_events_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_RISK_EVENTS_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "event": event,
                    "reason": reason,
                    "duration_hours": duration_hours,
                    "equity": round(equity, 2),
                })
        except Exception as exc:
            _log.error("_write_risk_event: %s", exc)

    # ── individual checks ──────────────────────────────────────────────────────

    def check_daily_loss(self, current_equity: float) -> Tuple[bool, str]:
        """Fail if today's realised P&L breaches MAX_DAILY_LOSS_PCT."""
        trades = self._read_today_trades()
        pnl_dia = sum(float(t.get("pnl", 0.0)) for t in trades)
        ratio = pnl_dia / (self.initial_capital + 1e-9)
        if ratio < -settings.MAX_DAILY_LOSS_PCT:
            reason = (
                f"daily_loss={ratio:.2%} > limit={settings.MAX_DAILY_LOSS_PCT:.2%} "
                f"(pnl_dia=${pnl_dia:.2f})"
            )
            _log.warning("check_daily_loss FAIL: %s", reason)
            return False, reason
        return True, ""

    def check_drawdown(self, current_equity: float) -> Tuple[bool, str]:
        """Fail if drawdown from equity_peak exceeds MAX_DRAWDOWN_PCT."""
        if current_equity > self.equity_peak:
            self.equity_peak = current_equity
        drawdown = (self.equity_peak - current_equity) / (self.equity_peak + 1e-9)
        if drawdown > settings.MAX_DRAWDOWN_PCT:
            reason = (
                f"drawdown={drawdown:.2%} > limit={settings.MAX_DRAWDOWN_PCT:.2%} "
                f"(peak=${self.equity_peak:.2f} current=${current_equity:.2f})"
            )
            _log.warning("check_drawdown FAIL: %s", reason)
            return False, reason
        return True, ""

    def check_consecutive_losses(self, symbol: str) -> Tuple[bool, str]:
        """
        Fail if the last MAX_CONSECUTIVE_LOSSES trades for *symbol* are all losses.

        Scans today's trade log first, then falls back to recent historical files.
        """
        max_n = settings.MAX_CONSECUTIVE_LOSSES
        recent: list[dict] = []

        # Collect enough rows (most-recent files first)
        trades_dir = Path(settings.LOGS_TRADES_DIR)
        if trades_dir.exists():
            for trade_file in sorted(trades_dir.glob("trades_*.csv"), reverse=True):
                try:
                    with open(trade_file, newline="", encoding="utf-8") as fh:
                        rows = list(csv.DictReader(fh))
                    symbol_rows = [r for r in rows if r.get("symbol") == symbol]
                    recent = symbol_rows + recent  # prepend older rows
                    if len(recent) >= max_n:
                        break
                except Exception as exc:
                    _log.warning("check_consecutive_losses: %s", exc)

        last_n = recent[-max_n:] if len(recent) >= max_n else recent
        if len(last_n) < max_n:
            return True, ""

        all_losses = all(float(r.get("pnl", 0.0)) < 0 for r in last_n)
        if all_losses:
            reason = (
                f"consecutive_losses={max_n} for {symbol}"
            )
            _log.warning("check_consecutive_losses FAIL: %s", reason)
            return False, reason
        return True, ""

    def check_margin(
        self,
        free_margin: float,
        margin: float,
    ) -> Tuple[bool, str]:
        """Fail if margin level is below MIN_MARGIN_BUFFER (or margin == 0)."""
        if margin <= 0:
            # No open positions — margin level is technically infinite, allow.
            return True, ""
        level = free_margin / margin
        if level < settings.MIN_MARGIN_BUFFER:
            reason = (
                f"margin_level={level:.2f} < min={settings.MIN_MARGIN_BUFFER:.2f} "
                f"(free_margin={free_margin:.2f} margin={margin:.2f})"
            )
            _log.warning("check_margin FAIL: %s", reason)
            return False, reason
        return True, ""

    def check_volatility(
        self,
        atr: float,
        atr_mean: float,
    ) -> Tuple[bool, str]:
        """Fail if current ATR is more than 2.5× the mean ATR (volatility spike)."""
        if atr_mean <= 0:
            return True, ""
        ratio = atr / (atr_mean + 1e-9)
        if ratio > 2.5:
            reason = (
                f"volatility spike: atr={atr:.5f} = {ratio:.2f}× atr_mean={atr_mean:.5f}"
            )
            _log.warning("check_volatility FAIL: %s", reason)
            return False, reason
        return True, ""

    def check_trading_hour(self) -> Tuple[bool, str]:
        """Fail if the current UTC hour is in NO_TRADE_HOURS_UTC."""
        utc_hour = datetime.now(tz=timezone.utc).hour
        if utc_hour in settings.NO_TRADE_HOURS_UTC:
            reason = f"no-trade hour UTC={utc_hour} (blocked hours: {settings.NO_TRADE_HOURS_UTC})"
            _log.debug("check_trading_hour FAIL: %s", reason)
            return False, reason
        return True, ""

    # ── aggregate check ────────────────────────────────────────────────────────

    def check_all(
        self,
        symbol: str,
        current_equity: float,
        atr: float,
        atr_mean: float,
        free_margin: float = 0.0,
        margin: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Run every risk check in priority order.

        Returns (True, "") if all checks pass, otherwise (False, reason) on the
        first failure.
        """
        checks = [
            self.check_trading_hour(),
            self.check_daily_loss(current_equity),
            self.check_drawdown(current_equity),
            self.check_consecutive_losses(symbol),
            self.check_volatility(atr, atr_mean),
            self.check_margin(free_margin, margin),
        ]
        for ok, reason in checks:
            if not ok:
                return False, reason
        return True, ""

    # ── emergency & pause ─────────────────────────────────────────────────────

    def emergency_stop(self, trade_manager, reason: str) -> None:
        """
        Close all open positions via *trade_manager* and log the emergency.

        *trade_manager* must expose a `close_all(reason: str)` method.
        """
        _log.critical("EMERGENCY STOP: %s", reason)
        notify(RISK_EVENT, {"reason": f"EMERGENCY STOP — {reason}"})
        try:
            trade_manager.close_all(reason=reason)
        except Exception as exc:
            _log.error("emergency_stop: trade_manager.close_all failed — %s", exc)

        # Persist emergency event with duration=0 (indefinite until manual review)
        self._write_risk_event(
            event="EMERGENCY_STOP",
            reason=reason,
            duration_hours=0.0,
            equity=0.0,  # equity unknown at this point; caller can pass if needed
        )

    def record_pause(self, reason: str, duration_hours: float) -> None:
        """Record a trading pause to risk_events.csv and notify."""
        _log.warning("RiskManager: PAUSE %.1fh — %s", duration_hours, reason)
        notify(RISK_EVENT, {"reason": f"PAUSE {duration_hours:.1f}h — {reason}"})
        self._write_risk_event(
            event="PAUSE",
            reason=reason,
            duration_hours=duration_hours,
            equity=0.0,
        )
