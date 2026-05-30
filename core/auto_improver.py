"""
Auto-mejora del bot basada en análisis de logs históricos.
Ajusta parámetros en runtime_params.json automáticamente.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.logger import get_logger
from config import settings
from analysis.log_analyzer import LogAnalyzer

_log = get_logger("auto_improver")


class AutoImprover:

    def __init__(self):
        self._analyzer = LogAnalyzer()
        self._params_path = settings.CONFIG_DIR / "runtime_params.json"

    def run(self, symbol: str) -> dict:
        """
        Analiza logs y ajusta parámetros si es necesario.
        Retorna dict con cambios realizados.
        """
        df = self._analyzer.load_all_trades(symbol)
        if len(df) < 30:
            _log.info("Insuficientes trades para auto-mejora (mínimo 30)")
            return {}

        changes = {}
        current = self._load_params()

        # Ventana de últimos 50 trades
        recent = df.tail(50) if len(df) >= 50 else df
        wr_recent = (recent["pnl_usd"] > 0).mean() if "pnl_usd" in recent.columns else 0

        wins   = recent[recent["pnl_usd"] > 0]["pnl_usd"] if "pnl_usd" in recent.columns else pd.Series()
        losses = recent[recent["pnl_usd"] <= 0]["pnl_usd"].abs() if "pnl_usd" in recent.columns else pd.Series()
        pf     = wins.sum() / (losses.sum() + 1e-9)

        _log.info(f"Auto-mejora {symbol}: WR={wr_recent:.1%} PF={pf:.2f} (últimos {len(recent)} trades)")

        # ── Ajuste de SIGNAL_THRESHOLD ─────────────────────────────────────────
        cur_thr = current.get("SIGNAL_THRESHOLD", settings.SIGNAL_THRESHOLD)
        if wr_recent < 0.50:
            new_thr = min(0.75, cur_thr + 0.03)
            changes["SIGNAL_THRESHOLD"] = new_thr
            _log.info(f"  WR muy bajo → threshold {cur_thr:.3f} → {new_thr:.3f}")
        elif wr_recent < 0.55:
            new_thr = min(0.73, cur_thr + 0.02)
            changes["SIGNAL_THRESHOLD"] = new_thr
        elif wr_recent > 0.68:
            new_thr = max(0.58, cur_thr - 0.01)
            changes["SIGNAL_THRESHOLD"] = new_thr
            _log.info(f"  WR excelente → bajando threshold para más trades: {cur_thr:.3f} → {new_thr:.3f}")

        # ── Ajuste de Kelly MAX según drawdown ─────────────────────────────────
        cur_kelly = current.get("KELLY_MAX_FRACTION", settings.KELLY_MAX_FRACTION)
        dd = self._calc_drawdown(df)
        if dd > 0.12:
            new_kelly = max(0.05, cur_kelly - 0.03)
            changes["KELLY_MAX_FRACTION"] = new_kelly
            _log.info(f"  Drawdown alto ({dd:.1%}) → Kelly max {cur_kelly:.2f} → {new_kelly:.2f}")
        elif pf > 2.0 and len(df) >= 100:
            new_kelly = min(0.25, cur_kelly + 0.02)
            changes["KELLY_MAX_FRACTION"] = new_kelly
            _log.info(f"  PF excelente → Kelly max {cur_kelly:.2f} → {new_kelly:.2f}")

        # ── Bloquear horas malas ───────────────────────────────────────────────
        worst_hours = self._analyzer.find_worst_hours(df, threshold=0.38)
        if worst_hours:
            cur_no_trade = current.get("NO_TRADE_HOURS_UTC", settings.NO_TRADE_HOURS_UTC)
            new_no_trade = sorted(set(cur_no_trade + worst_hours))
            if new_no_trade != cur_no_trade:
                changes["NO_TRADE_HOURS_UTC"] = new_no_trade
                _log.info(f"  Horas bloqueadas añadidas: {worst_hours}")

        # ── Guarda cambios ─────────────────────────────────────────────────────
        if changes:
            current.update(changes)
            self._save_params(current)
            self._write_report(symbol, changes, wr_recent, pf)

        return changes

    def needs_retrain(self, symbol: str) -> bool:
        """True si el win rate reciente cayó bajo 45% → re-entrenamiento urgente."""
        df = self._analyzer.load_all_trades(symbol)
        if len(df) < 50:
            return False
        recent = df.tail(50)
        wr = (recent["pnl_usd"] > 0).mean() if "pnl_usd" in recent.columns else 1.0
        if wr < 0.45:
            _log.warning(f"WR reciente {wr:.1%} < 45% — re-entrenamiento necesario")
            return True
        return False

    def _calc_drawdown(self, df: pd.DataFrame) -> float:
        if "capital" not in df.columns or df.empty:
            return 0.0
        equity = df["capital"].values
        peak   = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def _load_params(self) -> dict:
        if self._params_path.exists():
            try:
                return json.loads(self._params_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_params(self, params: dict):
        tmp = self._params_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(params, indent=2))
        tmp.replace(self._params_path)
        _log.info(f"runtime_params.json actualizado")

    def _write_report(self, symbol: str, changes: dict, wr: float, pf: float):
        path = settings.LOGS_SYSTEM_DIR / \
               f"improvements_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt"
        settings.LOGS_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Auto-Improver Report — {symbol} — {datetime.utcnow().isoformat()}",
            f"WR reciente: {wr:.1%} | PF: {pf:.2f}",
            "Cambios aplicados:",
        ] + [f"  {k}: {v}" for k, v in changes.items()]
        path.write_text("\n".join(lines))
