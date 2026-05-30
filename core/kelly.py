"""
KellyEngine — Position sizing via Kelly Criterion.

Persists trade statistics and computed fractions to:
    logs/system/kelly_history.csv
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import settings
from utils.logger import get_logger

_log = get_logger("kelly")

_HISTORY_COLS = ["timestamp", "symbol", "fraction", "win_rate", "avg_win", "avg_loss"]


class KellyEngine:
    """
    Calculates Kelly fractions and converts them to lot sizes.

    Internal per-symbol stats are kept in memory and persisted to
    kelly_history.csv on every recalculate() call.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        # {symbol: {"wins": int, "losses": int, "win_pips": float, "loss_pips": float}}
        self._stats: Dict[str, dict] = {}
        self._history_path = Path(settings.LOGS_SYSTEM_DIR) / "kelly_history.csv"
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    # ── private helpers ────────────────────────────────────────────────────────

    def _ensure_stats(self, symbol: str) -> None:
        if symbol not in self._stats:
            self._stats[symbol] = {
                "wins": 0,
                "losses": 0,
                "win_pips": 0.0,
                "loss_pips": 0.0,
            }

    def _trade_count(self, symbol: str) -> int:
        s = self._stats.get(symbol, {})
        return s.get("wins", 0) + s.get("losses", 0)

    def _load_history(self) -> None:
        """Reconstruct internal stats from the last row in kelly_history.csv."""
        if not self._history_path.exists():
            return
        try:
            last_rows: Dict[str, dict] = {}
            with open(self._history_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    last_rows[row["symbol"]] = row
            for sym, row in last_rows.items():
                # Reconstruct minimal stats so win_rate / avg_win / avg_loss are
                # consistent on cold start (we store aggregated values, not raw
                # counts, so we synthesise from stored averages).
                win_rate = float(row.get("win_rate", settings.KELLY_BOOTSTRAP_FRAC))
                avg_win = float(row.get("avg_win", 0.0))
                avg_loss = float(row.get("avg_loss", 0.0))
                # Use KELLY_RECALC_EVERY as synthetic total so the engine knows
                # it already has "enough" history to exit bootstrap mode.
                synthetic_total = settings.KELLY_RECALC_EVERY
                wins = int(round(win_rate * synthetic_total))
                losses = synthetic_total - wins
                self._stats[sym] = {
                    "wins": wins,
                    "losses": losses,
                    "win_pips": avg_win * wins if wins else 0.0,
                    "loss_pips": avg_loss * losses if losses else 0.0,
                }
            _log.info("KellyEngine: loaded history for symbols: %s", list(last_rows.keys()))
        except Exception as exc:
            _log.warning("KellyEngine: could not load kelly_history.csv — %s", exc)

    def _append_history(
        self,
        symbol: str,
        fraction: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> None:
        write_header = not self._history_path.exists() or self._history_path.stat().st_size == 0
        try:
            with open(self._history_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_HISTORY_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "fraction": round(fraction, 6),
                    "win_rate": round(win_rate, 6),
                    "avg_win": round(avg_win, 4),
                    "avg_loss": round(avg_loss, 4),
                })
        except Exception as exc:
            _log.error("KellyEngine: failed to write kelly_history.csv — %s", exc)

    # ── public API ─────────────────────────────────────────────────────────────

    def calculate_fraction(
        self,
        win_rate: float,
        avg_win_pips: float,
        avg_loss_pips: float,
    ) -> float:
        """
        Full Kelly formula.

            b     = avg_win_pips / avg_loss_pips
            f*    = (b * p - q) / b
            where p = win_rate, q = 1 - win_rate

        Result is clamped to [KELLY_MIN_FRACTION, KELLY_MAX_FRACTION].
        """
        b = avg_win_pips / (avg_loss_pips + 1e-9)
        f_star = (b * win_rate - (1.0 - win_rate)) / (b + 1e-9)
        clamped = max(settings.KELLY_MIN_FRACTION, min(settings.KELLY_MAX_FRACTION, f_star))
        _log.debug(
            "calculate_fraction: win_rate=%.3f b=%.3f f*=%.4f clamped=%.4f",
            win_rate, b, f_star, clamped,
        )
        return clamped

    def fraction_to_lots(
        self,
        fraction: float,
        capital: float,
        symbol: str,
        price: float,
    ) -> float:
        """Convert a Kelly fraction to a lot size, respecting min/max bounds."""
        leverage = (
            settings.LEVERAGE_PHASE2 if "XAU" in symbol else settings.LEVERAGE_PHASE1
        )
        notional = capital * fraction * leverage

        if "XAU" in symbol:
            # 1 XAU lot = 100 oz
            lots = notional / (price * 100)
        else:
            # 1 standard FX lot = 100,000 units
            lots = notional / 100_000

        lots = round(lots, 2)
        lots = max(settings.KELLY_MIN_LOTS, min(settings.KELLY_MAX_LOTS, lots))
        _log.debug(
            "fraction_to_lots: symbol=%s fraction=%.4f capital=%.2f price=%.5f "
            "leverage=%d notional=%.2f lots=%.2f",
            symbol, fraction, capital, price, leverage, notional, lots,
        )
        return lots

    def verify_margin(
        self,
        lots: float,
        symbol: str,
        free_margin: float,
        symbol_info: dict,
    ) -> bool:
        """
        Returns True if free_margin covers the required margin with a 50% buffer.

        margin_required = lots * contract_size / leverage
        """
        leverage = (
            settings.LEVERAGE_PHASE2 if "XAU" in symbol else settings.LEVERAGE_PHASE1
        )
        contract_size = symbol_info.get("trade_contract_size", 100_000)
        margin_required = (lots * contract_size) / leverage
        sufficient = free_margin >= margin_required * 1.5
        if not sufficient:
            _log.warning(
                "verify_margin FAIL: symbol=%s lots=%.2f free_margin=%.2f "
                "margin_required=%.2f (x1.5=%.2f)",
                symbol, lots, free_margin, margin_required, margin_required * 1.5,
            )
        return sufficient

    def get_current_fraction(self, symbol: str) -> float:
        """
        Return the current Kelly fraction for *symbol*.

        Uses bootstrap fraction when fewer than KELLY_RECALC_EVERY trades are
        recorded (cold start / new symbol).
        """
        self._ensure_stats(symbol)
        n = self._trade_count(symbol)
        if n < settings.KELLY_RECALC_EVERY:
            _log.debug(
                "get_current_fraction: %s bootstrap (n=%d < %d)",
                symbol, n, settings.KELLY_RECALC_EVERY,
            )
            return settings.KELLY_BOOTSTRAP_FRAC

        s = self._stats[symbol]
        win_rate = s["wins"] / n
        avg_win = s["win_pips"] / s["wins"] if s["wins"] else 0.0
        avg_loss = s["loss_pips"] / s["losses"] if s["losses"] else 1e-9
        return self.calculate_fraction(win_rate, avg_win, avg_loss)

    def update(self, symbol: str, win: bool, pips_result: float) -> None:
        """Record the outcome of a closed trade and update running statistics."""
        self._ensure_stats(symbol)
        s = self._stats[symbol]
        if win:
            s["wins"] += 1
            s["win_pips"] += abs(pips_result)
        else:
            s["losses"] += 1
            s["loss_pips"] += abs(pips_result)
        _log.debug(
            "update: symbol=%s win=%s pips=%.1f total_trades=%d",
            symbol, win, pips_result, self._trade_count(symbol),
        )

    def recalculate(self, symbol: str) -> float:
        """
        Recompute f* from current stats and persist to kelly_history.csv.

        Returns the new fraction.
        """
        self._ensure_stats(symbol)
        n = self._trade_count(symbol)
        if n < settings.KELLY_RECALC_EVERY:
            fraction = settings.KELLY_BOOTSTRAP_FRAC
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        else:
            s = self._stats[symbol]
            win_rate = s["wins"] / n
            avg_win = s["win_pips"] / s["wins"] if s["wins"] else 0.0
            avg_loss = s["loss_pips"] / s["losses"] if s["losses"] else 1e-9
            fraction = self.calculate_fraction(win_rate, avg_win, avg_loss)

        self._append_history(symbol, fraction, win_rate, avg_win, avg_loss)
        _log.info(
            "recalculate: symbol=%s n=%d win_rate=%.3f avg_win=%.2f "
            "avg_loss=%.2f fraction=%.4f",
            symbol, n, win_rate, avg_win, avg_loss, fraction,
        )
        return fraction
