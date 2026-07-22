"""
Motor de backtesting event-driven sobre velas OHLC historicas.

IMPORTANTE (leer antes de confiar en cualquier resultado):
  - Este modulo NO viene con datos historicos embebidos ni resultados
    precalculados. Para backtestear de verdad necesitas pasarle un DataFrame
    con >= 6 meses de velas M1 reales de XAUUSD (exportadas desde MT5 via
    `mt5_connector.get_rates()` con rango amplio, o un CSV propio con columnas
    time/open/high/low/close/tick_volume).
  - `SyntheticSymbolInfo` trae valores tipicos de XAUUSD (contrato 100 oz,
    tick 0.01) SOLO como placeholder. Antes de confiar en las metricas, reemplazalos
    por la especificacion real de tu broker (`mt5.symbol_info("XAUUSD")`),
    porque tick_value/spread/stops_level cambian el resultado de forma
    significativa.
  - Simplificacion de alcance: el backtester simula UNA posicion logica a la
    vez (que puede escalar via grid), no multiples posiciones concurrentes
    independientes. En vivo, `main.py` si soporta hasta MAX_OPEN_POSITIONS
    simultaneas reales via MT5. Esto mantiene el motor tratable sin inflar
    artificialmente las metricas con posiciones paralelas correlacionadas.
  - El relleno de ordenes usa el `open` de la vela SIGUIENTE a la señal (evita
    look-ahead bias), y SL/TP intrabar se evaluan contra el high/low de cada
    vela subsiguiente (aproximacion OHLC, no simulacion a nivel de tick).
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import config
import grid_trader
import profit_manager as profit_manager_module
import risk_management
import stop_loss_calculator
from indicators import atr as atr_indicator
from strategies import confluence_engine

logger = logging.getLogger(__name__)


@dataclass
class SyntheticSymbolInfo:
    """Placeholder de especificacion de simbolo para backtesting offline.
    Reemplazar por mt5.symbol_info(config.SYMBOL) real cuando esta disponible."""
    point: float = 0.01
    digits: int = 2
    trade_stops_level: float = 0.0
    trade_tick_value: float = 1.0
    trade_tick_size: float = 0.01
    trade_contract_size: float = 100.0
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 50.0


@dataclass
class BacktestTrade:
    entry_time: object
    exit_time: object
    direction: int
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    reason: str


@dataclass
class BacktestReport:
    trades: List[BacktestTrade]
    equity_curve: List[float]
    initial_balance: float
    final_balance: float

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in self.trades if t.pnl < 0)
        return gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        return float(np.max(drawdown) * 100.0)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe aproximado sobre retornos por trade (no anualizado). Solo
        orientativo: con pocas muestras o alta autocorrelacion de HFT-scalping
        este numero es ruidoso, no debe tomarse como metrica unica de decision."""
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl for t in self.trades]) / self.initial_balance
        std = returns.std()
        if std == 0:
            return 0.0
        return float(returns.mean() / std * np.sqrt(len(returns)))

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Win rate: {self.win_rate:.1%} | "
            f"Profit factor: {self.profit_factor:.2f} | Max DD: {self.max_drawdown_pct:.2f}% | "
            f"Sharpe (por trade, no anualizado): {self.sharpe_ratio:.2f} | "
            f"Balance: {self.initial_balance:.2f} -> {self.final_balance:.2f}"
        )


def run_backtest(df: pd.DataFrame, initial_balance: float = 1000.0,
                  symbol_info: Optional[SyntheticSymbolInfo] = None,
                  spread_points: float = 20.0) -> BacktestReport:
    symbol_info = symbol_info or SyntheticSymbolInfo()
    df = df.reset_index(drop=True)
    min_window = max(config.EMA_TREND, config.BOLLINGER_PERIOD, config.MACD_SLOW + config.MACD_SIGNAL) + 10

    balance = initial_balance
    equity_curve = [balance]
    trades: List[BacktestTrade] = []

    position = None  # dict con estado de la posicion logica abierta
    pm = profit_manager_module.ProfitManager()
    risk_mgr = risk_management.RiskManager()

    atr_full = atr_indicator(df, config.ATR_PERIOD)

    i = min_window
    while i < len(df) - 1:
        window = df.iloc[max(0, i - config.BARS_LOOKBACK): i + 1]
        bar_next = df.iloc[i + 1]
        current_atr = atr_full.iloc[i]

        if position is None:
            can_trade, _ = risk_mgr.can_open_new_trade(balance, 0)
            if can_trade and current_atr > 0:
                result = confluence_engine.evaluate_confluence(window)
                if result.is_actionable:
                    entry_price = bar_next["open"]
                    plan = stop_loss_calculator.calculate_stop_loss_plan(
                        int(result.direction), entry_price, window, symbol_info, spread_points,
                    )
                    if plan.valid:
                        risk_capital = balance * (config.RISK_PER_TRADE_PCT / 100.0)
                        volume = risk_management.calculate_position_size(risk_capital, plan.sl_distance_price, symbol_info)
                        if volume > 0:
                            position = {
                                "direction": int(result.direction),
                                "entry_price": entry_price,
                                "entry_time": bar_next.get("time", i + 1),
                                "volume": volume,
                                "initial_volume": volume,
                                "sl": plan.sl_price,
                                "tp1": plan.tp1_price,
                                "tp2": plan.tp2_price,
                            }
                            pm.register_position(0, entry_price, position["direction"], volume,
                                                  plan.tp1_price, plan.tp2_price)
                            risk_mgr.register_trade_opened()
            i += 1
            continue

        # --- Gestionar posicion abierta contra la vela siguiente ---------
        direction = position["direction"]
        hit_sl = (bar_next["low"] <= position["sl"]) if direction == 1 else (bar_next["high"] >= position["sl"])

        if hit_sl:
            exit_price = position["sl"]
            pnl = (exit_price - position["entry_price"]) * direction * position["volume"] * symbol_info.trade_contract_size
            balance += pnl
            trades.append(BacktestTrade(position["entry_time"], bar_next.get("time", i + 1), direction,
                                         position["entry_price"], exit_price, position["volume"], pnl, "stop loss"))
            pm.forget_position(0)
            position = None
            equity_curve.append(balance)
            i += 1
            continue

        current_price = bar_next["close"]
        actions = pm.evaluate(0, current_price, position["sl"], current_atr)

        for action in actions:
            if action.type == profit_manager_module.ActionType.PARTIAL_CLOSE:
                vol = min(action.volume, position["volume"])
                pnl = (current_price - position["entry_price"]) * direction * vol * symbol_info.trade_contract_size
                balance += pnl
                position["volume"] = round(position["volume"] - vol, 2)
                trades.append(BacktestTrade(position["entry_time"], bar_next.get("time", i + 1), direction,
                                             position["entry_price"], current_price, vol, pnl, action.reason))
            elif action.type == profit_manager_module.ActionType.MODIFY_SL:
                position["sl"] = action.sl_price
            elif action.type == profit_manager_module.ActionType.FULL_CLOSE:
                pnl = (current_price - position["entry_price"]) * direction * position["volume"] * symbol_info.trade_contract_size
                balance += pnl
                trades.append(BacktestTrade(position["entry_time"], bar_next.get("time", i + 1), direction,
                                             position["entry_price"], current_price, position["volume"], pnl, action.reason))
                position = None

        if position is not None and position["volume"] <= 0:
            position = None

        equity_curve.append(balance)
        i += 1

    return BacktestReport(trades=trades, equity_curve=equity_curve,
                           initial_balance=initial_balance, final_balance=balance)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python backtester.py <ruta_csv_con_columnas_time,open,high,low,close,tick_volume>")
        sys.exit(1)

    data = pd.read_csv(sys.argv[1])
    report = run_backtest(data)
    print(report.summary())
