"""
Grid search + walk-forward para optimización de parámetros.
"""
import json
from itertools import product
from pathlib import Path
import pandas as pd

from utils.logger import get_logger
from config import settings

_log = get_logger("optimizer")


class Optimizer:

    GRID = {
        "SIGNAL_THRESHOLD":   [0.58, 0.60, 0.62, 0.65, 0.68],
        "ATR_SL_MULTIPLIER":  [1.2, 1.5, 1.8],
        "ATR_TP1_MULTIPLIER": [1.8, 2.0, 2.5],
    }

    def optimize(self, symbol: str, df_features: pd.DataFrame,
                 feature_cols: list, model, backtester_cls,
                 initial_capital: float = settings.INITIAL_CAPITAL) -> dict:
        """
        Grid search sobre parámetros con walk-forward validation.
        Métrica: Profit Factor con restricción MaxDD < 20%.
        """
        n      = len(df_features)
        n_val  = max(200, int(n * 0.30))
        df_opt = df_features.iloc[:-n_val].copy()
        df_val = df_features.iloc[-n_val:].copy()

        best_params = {}
        best_pf     = 0.0
        results     = []

        total = len(self.GRID["SIGNAL_THRESHOLD"]) * len(self.GRID["ATR_SL_MULTIPLIER"]) * \
                len(self.GRID["ATR_TP1_MULTIPLIER"])
        _log.info(f"Grid search: {total} combinaciones para {symbol}")

        for thr, sl_mult, tp1_mult in product(
                self.GRID["SIGNAL_THRESHOLD"],
                self.GRID["ATR_SL_MULTIPLIER"],
                self.GRID["ATR_TP1_MULTIPLIER"]):

            # Aplica parámetros temporalmente
            orig_thr  = settings.SIGNAL_THRESHOLD
            orig_sl   = settings.ATR_SL_MULTIPLIER
            orig_tp1  = settings.ATR_TP1_MULTIPLIER
            settings.SIGNAL_THRESHOLD   = thr
            settings.ATR_SL_MULTIPLIER  = sl_mult
            settings.ATR_TP1_MULTIPLIER = tp1_mult

            try:
                # Backtest en opt set
                bt_opt = backtester_cls(symbol)
                m_opt  = bt_opt.run(df_opt, model, feature_cols, thr, initial_capital)

                # Backtest en val set (walk-forward)
                bt_val = backtester_cls(symbol)
                m_val  = bt_val.run(df_val, model, feature_cols, thr, initial_capital)
            finally:
                settings.SIGNAL_THRESHOLD   = orig_thr
                settings.ATR_SL_MULTIPLIER  = orig_sl
                settings.ATR_TP1_MULTIPLIER = orig_tp1

            # Filtros anti-overfitting
            if m_opt.get("total_trades", 0) < 30:
                continue
            if m_opt.get("max_drawdown", 1) > 0.20:
                continue
            if m_val.get("total_trades", 0) < 10:
                continue

            # Anti-overfitting: diferencia WR < 25%
            wr_diff = abs(m_opt.get("win_rate", 0) - m_val.get("win_rate", 0))
            if wr_diff > 0.25:
                continue

            pf_val = m_val.get("profit_factor", 0)
            results.append({
                "threshold":  thr,
                "sl_mult":    sl_mult,
                "tp1_mult":   tp1_mult,
                "pf_val":     pf_val,
                "wr_val":     m_val.get("win_rate", 0),
                "pf_opt":     m_opt.get("profit_factor", 0),
                "wr_opt":     m_opt.get("win_rate", 0),
                "wr_diff":    wr_diff,
            })

            if pf_val > best_pf:
                best_pf     = pf_val
                best_params = {
                    "SIGNAL_THRESHOLD":   thr,
                    "ATR_SL_MULTIPLIER":  sl_mult,
                    "ATR_TP1_MULTIPLIER": tp1_mult,
                }

        if best_params:
            self._save(best_params)
            _log.info(f"Mejores parámetros: {best_params} (PF_val={best_pf:.2f})")

        return {"best_params": best_params, "best_pf": best_pf, "results": results}

    def _save(self, params: dict):
        path = settings.CONFIG_DIR / "optimized_params.json"
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update(params)
            path.write_text(json.dumps(current, indent=2))
        except Exception as e:
            _log.warning(f"No se pudo guardar optimized_params: {e}")
