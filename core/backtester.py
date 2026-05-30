"""
Backtester vela a vela estilo MT5.
Incluye simulación de spread, slippage, trailing SL y TP parcial.
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
        Simula trading vela a vela sobre df (set de test).
        Retorna dict con métricas completas.
        """
        self._results = []
        capital       = initial_capital
        open_trade    = None
        equity_curve  = [capital]

        # Spread y slippage simulados
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

            # ── Gestión de trade abierto ─────────────────────────────────────
            if open_trade is not None:
                open_trade["bars_open"] += 1
                exit_result = self._check_exit(open_trade, bar, atr)
                if exit_result:
                    pnl, reason, exit_price = exit_result
                    capital += pnl
                    self._record_trade(open_trade, exit_price, pnl, reason, capital)
                    equity_curve.append(capital)
                    open_trade = None
                    continue

            # ── Nueva señal ──────────────────────────────────────────────────
            if open_trade is not None:
                continue

            if i + 1 >= len(df):
                break

            # Features de la vela actual
            try:
                x = df[feature_cols].iloc[i:i+1].fillna(0)
                proba_buy = float(model.predict_proba(x)[0, 1])
            except Exception:
                continue

            direction, proba = self._get_direction(proba_buy)
            if proba < signal_threshold:
                continue

            # Filtro de régimen si está disponible
            if regime_detector is not None:
                regime, _ = regime_detector.detect(df.iloc[max(0, i-100):i+1])
                if regime in (constants.REGIME_NO_TRADE, constants.REGIME_VOLATILE):
                    continue
            else:
                # Filtro horario básico
                try:
                    hour = pd.to_datetime(row.get("time", 0)).hour
                    if hour in settings.NO_TRADE_HOURS_UTC:
                        continue
                except Exception:
                    pass

            # Entrada al open de la vela siguiente + slippage
            if i + 1 >= len(df):
                break
            next_bar = df.iloc[i + 1]
            slippage = random.uniform(0, 0.5) * point
            entry    = next_bar["open"] + (slippage if direction == constants.SIGNAL_BUY
                                           else -slippage)
            # Añadir spread al BUY
            if direction == constants.SIGNAL_BUY:
                entry += spread_f

            # Kelly position sizing
            lots = self._calc_lots(capital, atr)
            if lots <= 0:
                continue

            # Niveles SL/TP
            sl  = entry - atr * settings.ATR_SL_MULTIPLIER  if direction == constants.SIGNAL_BUY \
                  else entry + atr * settings.ATR_SL_MULTIPLIER
            tp1 = entry + atr * settings.ATR_TP1_MULTIPLIER if direction == constants.SIGNAL_BUY \
                  else entry - atr * settings.ATR_TP1_MULTIPLIER
            tp2 = entry + atr * settings.ATR_TP2_MULTIPLIER if direction == constants.SIGNAL_BUY \
                  else entry - atr * settings.ATR_TP2_MULTIPLIER

            # Valor pip
            pip_value = self._pip_value(lots, self.symbol)

            open_trade = {
                "entry":      entry,
                "direction":  direction,
                "lots":       lots,
                "sl":         sl,
                "tp1":        tp1,
                "tp2":        tp2,
                "phase":      1,
                "trailing_sl": None,
                "partial_done": False,
                "bars_open":   0,
                "open_time":   row.get("time", i),
                "pip_value":   pip_value,
                "atr_entry":  atr,
                "proba":       proba,
            }

        # Cierra trade abierto al final
        if open_trade is not None:
            last = df.iloc[-1]
            pnl  = self._calc_pnl(open_trade, last["close"])
            capital += pnl
            self._record_trade(open_trade, last["close"], pnl, "END_OF_DATA", capital)
            equity_curve.append(capital)

        return self._compute_metrics(initial_capital, capital, equity_curve)

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, trade: dict, bar: dict, atr: float):
        is_buy   = trade["direction"] == constants.SIGNAL_BUY
        high, low, close = bar["high"], bar["low"], bar["close"]

        # SL hit
        if is_buy and low <= trade["sl"]:
            pnl = self._calc_pnl(trade, trade["sl"])
            return pnl, constants.EXIT_SL, trade["sl"]
        if not is_buy and high >= trade["sl"]:
            pnl = self._calc_pnl(trade, trade["sl"])
            return pnl, constants.EXIT_SL, trade["sl"]

        # Fase 1: TP1
        if trade["phase"] == 1:
            if (is_buy and high >= trade["tp1"]) or (not is_buy and low <= trade["tp1"]):
                # Cierra 50%, actualiza trade para fase 2
                spread_adj = atr * 0.05
                trade["sl"]          = trade["entry"] + spread_adj if is_buy else trade["entry"] - spread_adj
                trade["lots"]       *= (1 - settings.PARTIAL_TP_FRACTION)
                trade["lots"]        = max(trade["lots"], settings.KELLY_MIN_LOTS)
                trade["phase"]       = 2
                trade["trailing_sl"] = trade["sl"]
                trade["partial_done"] = True
                # No cerramos completamente, solo actualizamos estado
                return None

        # Fase 2: trailing
        if trade["phase"] == 2:
            trail_dist = atr * settings.ATR_TRAILING_MULTIPLIER
            if is_buy:
                new_trail = close - trail_dist
                if new_trail > trade.get("trailing_sl") or trade["trailing_sl"] is None:
                    trade["trailing_sl"] = max(new_trail, trade["sl"])
                if high >= trade["tp2"]:
                    pnl = self._calc_pnl(trade, trade["tp2"])
                    return pnl, constants.EXIT_TP2, trade["tp2"]
                if low <= trade["trailing_sl"]:
                    pnl = self._calc_pnl(trade, trade["trailing_sl"])
                    return pnl, constants.EXIT_TRAILING, trade["trailing_sl"]
            else:
                new_trail = close + trail_dist
                if new_trail < trade.get("trailing_sl") or trade["trailing_sl"] is None:
                    trade["trailing_sl"] = min(new_trail, trade["sl"])
                if low <= trade["tp2"]:
                    pnl = self._calc_pnl(trade, trade["tp2"])
                    return pnl, constants.EXIT_TP2, trade["tp2"]
                if high >= trade["trailing_sl"]:
                    pnl = self._calc_pnl(trade, trade["trailing_sl"])
                    return pnl, constants.EXIT_TRAILING, trade["trailing_sl"]

        # Time exit
        if trade["bars_open"] >= 30:
            min_prog = atr * 0.2
            progress = (close - trade["entry"]) if is_buy else (trade["entry"] - close)
            if progress < min_prog:
                pnl = self._calc_pnl(trade, close)
                return pnl, constants.EXIT_TIME, close

        return None

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
        leverage = settings.LEVERAGE_PHASE1 if "XAU" not in self.symbol else settings.LEVERAGE_PHASE2
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
            "tp1":         trade["tp1"],
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

        # Sharpe (diario, simplificado)
        returns = df["pnl_usd"].values
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0

        # Max drawdown
        peak    = initial_capital
        max_dd  = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Guarda CSV
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
