"""
Backtester vela a vela estilo MT5.
Ciclo de 1 vela: cada trade se abre y se cierra en la vela siguiente a la
señal — sin TP2, sin trailing, sin hold multi-vela. Simula spread y slippage.
"""
import random
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("backtester")

random.seed(42)


class Backtester:
    def __init__(self, symbol: str):
        self.symbol   = symbol
        self._results: list = []

    def run(self, df: pd.DataFrame, model, feature_cols: list,
            signal_threshold: float, initial_capital: float,
            regime_detector=None, mtf_analyzer=None) -> dict:
        """
        Simula trading vela a vela sobre df (set de test). Un trade por vela:
        se fuerza el cierre del trade abierto y de inmediato se evalúa/abre
        el siguiente, sin huecos. signal_threshold/regime_detector/mtf_analyzer
        se aceptan por compatibilidad de firma pero ya no filtran señales —
        el único filtro es el spread simulado vs MAX_SPREAD_USD.
        Retorna dict con métricas completas.
        """
        self._results = []
        capital       = initial_capital
        open_trade    = None
        equity_curve  = [capital]

        is_xau   = "XAU" in self.symbol
        spread   = (settings.SPREAD_POINTS_XAUUSD if is_xau else settings.SPREAD_POINTS_EURUSD)
        point    = 0.01 if is_xau else 0.00001
        spread_f = spread * point

        for i in range(len(df)):
            row  = df.iloc[i]
            bar  = {"open": row["open"], "high": row["high"],
                    "low": row["low"],   "close": row["close"],
                    "time": row.get("time", i)}
            atr  = float(row["atr"]) if "atr" in row else spread_f * 15

            # ── Cierre forzado del trade abierto en la vela anterior ─────────
            if open_trade is not None:
                open_trade["bars_open"] += 1
                pnl, reason, exit_price = self._check_exit(open_trade, bar, atr)
                capital += pnl
                self._record_trade(open_trade, exit_price, pnl, reason, capital)
                equity_curve.append(capital)
                open_trade = None

            if i + 1 >= len(df):
                break

            # ── Único filtro: spread por encima del máximo tolerado ──────────
            if spread_f > settings.MAX_SPREAD_USD:
                continue

            try:
                x = df[feature_cols].iloc[i:i+1].fillna(0)
                proba_buy = float(model.predict_proba(x)[0, 1])
            except Exception:
                continue

            direction, proba = self._get_direction(proba_buy)

            next_bar = df.iloc[i + 1]
            slippage = random.uniform(0, 0.5) * point
            entry    = next_bar["open"] + (slippage if direction == constants.SIGNAL_BUY
                                           else -slippage)
            if direction == constants.SIGNAL_BUY:
                entry += spread_f

            lots = self._calc_lots(capital, atr)
            if lots <= 0:
                continue

            sl = entry - atr * settings.ATR_SL_MULTIPLIER  if direction == constants.SIGNAL_BUY \
                 else entry + atr * settings.ATR_SL_MULTIPLIER
            tp = entry + atr * settings.ATR_TP1_MULTIPLIER if direction == constants.SIGNAL_BUY \
                 else entry - atr * settings.ATR_TP1_MULTIPLIER

            pip_value = self._pip_value(lots, self.symbol)

            open_trade = {
                "entry":      entry,
                "direction":  direction,
                "lots":       lots,
                "sl":         sl,
                "tp":         tp,
                "bars_open":  0,
                "open_time":  row.get("time", i),
                "pip_value":  pip_value,
                "atr_entry":  atr,
                "proba":      proba,
            }

        if open_trade is not None:
            last = df.iloc[-1]
            pnl  = self._calc_pnl(open_trade, last["close"])
            capital += pnl
            self._record_trade(open_trade, last["close"], pnl, constants.EXIT_BAR_CLOSE, capital)
            equity_curve.append(capital)

        return self._compute_metrics(initial_capital, capital, equity_curve)

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, trade: dict, bar: dict, atr: float):
        """Siempre devuelve un resultado: SL, TP, o cierre forzado al close de la vela."""
        is_buy = trade["direction"] == constants.SIGNAL_BUY
        high, low, close = bar["high"], bar["low"], bar["close"]

        if is_buy and low <= trade["sl"]:
            return self._calc_pnl(trade, trade["sl"]), constants.EXIT_SL, trade["sl"]
        if not is_buy and high >= trade["sl"]:
            return self._calc_pnl(trade, trade["sl"]), constants.EXIT_SL, trade["sl"]

        tp_hit = (is_buy and high >= trade["tp"]) or (not is_buy and low <= trade["tp"])
        if tp_hit:
            return self._calc_pnl(trade, trade["tp"]), constants.EXIT_TP, trade["tp"]

        return self._calc_pnl(trade, close), constants.EXIT_BAR_CLOSE, close

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_direction(proba_buy: float):
        if proba_buy >= 0.5:
            return constants.SIGNAL_BUY, proba_buy
        return constants.SIGNAL_SELL, 1.0 - proba_buy

    @staticmethod
    def _pip_value(lots: float, symbol: str) -> float:
        if "XAU" in symbol:
            return lots * 100 * 0.1   # 1 lot = 100oz, pip = $0.1/oz
        return lots * 100000 * 0.0001  # 1 lot = 100k EUR, pip = $10

    def _calc_pnl(self, trade: dict, exit_price: float) -> float:
        diff = exit_price - trade["entry"] if trade["direction"] == constants.SIGNAL_BUY \
               else trade["entry"] - exit_price
        pip_div = 0.1 if "XAU" in self.symbol else 0.00010
        pips    = diff / pip_div
        return round(pips * trade["pip_value"], 4)

    def _calc_lots(self, capital: float, atr: float) -> float:
        fraction = settings.KELLY_BOOTSTRAP_FRAC
        leverage = settings.LEVERAGE_XAUUSD
        notional = capital * fraction * leverage
        if "XAU" in self.symbol:
            lots = notional / (2300 * 100)
        else:
            lots = notional / 100000
        lots = round(lots, 2)
        return max(settings.KELLY_MIN_LOTS, min(settings.KELLY_MAX_LOTS, lots))

    def _record_trade(self, trade: dict, exit_price: float, pnl: float, reason: str, capital: float):
        pip_div = 0.1 if "XAU" in self.symbol else 0.00010
        diff    = exit_price - trade["entry"] if trade["direction"] == constants.SIGNAL_BUY \
                  else trade["entry"] - exit_price
        pips    = round(diff / pip_div, 1)
        self._results.append({
            "open_time":   trade["open_time"],
            "symbol":      self.symbol,
            "direction":   trade["direction"],
            "lots":        trade["lots"],
            "entry":       trade["entry"],
            "exit_price":  exit_price,
            "sl":          trade["sl"],
            "tp":          trade["tp"],
            "pips":        pips,
            "pnl_usd":     pnl,
            "capital":     round(capital, 4),
            "reason":      reason,
            "bars_open":   trade["bars_open"],
            "proba":       trade.get("proba", 0),
        })

    def _compute_metrics(self, initial_capital: float, final_capital: float,
                          equity_curve: list) -> dict:
        if not self._results:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "net_pnl": 0, "final_capital": final_capital,
                    "max_drawdown": 0, "sharpe": 0, "roi": 0,
                    "equity_curve": equity_curve}

        df    = pd.DataFrame(self._results)
        wins  = df[df["pnl_usd"] > 0]
        loses = df[df["pnl_usd"] <= 0]

        win_rate      = len(wins) / len(df) if len(df) > 0 else 0
        sum_wins      = wins["pnl_usd"].sum()
        sum_losses    = loses["pnl_usd"].abs().sum()
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else float("inf")
        net_pnl       = final_capital - initial_capital
        roi           = net_pnl / initial_capital

        returns = df["pnl_usd"].values
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0

        peak    = initial_capital
        max_dd  = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        self._save_csv(df)

        return {
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(profit_factor, 3),
            "total_trades":  len(df),
            "wins":          len(wins),
            "losses":        len(loses),
            "net_pnl":       round(net_pnl, 4),
            "final_capital": round(final_capital, 4),
            "max_drawdown":  round(max_dd, 4),
            "sharpe":        round(float(sharpe), 3),
            "roi":           round(roi, 4),
            "avg_win_pips":  round(wins["pips"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loss_pips": round(loses["pips"].abs().mean(), 2) if len(loses) > 0 else 0,
            "equity_curve":  equity_curve,
            "trades_df":     df,
        }

    def _save_csv(self, df: pd.DataFrame):
        path = settings.LOGS_BACKTEST_DIR / \
               f"backtest_{self.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
        settings.LOGS_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["trades_df"] if "trades_df" in df.columns else [], errors="ignore") \
          .to_csv(path, index=False)
        _log.info(f"Backtest guardado: {path.name}")
