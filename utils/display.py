import os
import sys
from typing import List, Any
from config.constants import BOT_VERSION

_W = 54

def _line(char="═") -> str:
    return char * _W

def print_banner():
    print(f"\n{'╔' + _line() + '╗'}")
    print(f"{'║'}{f'  BOT-Trading v{BOT_VERSION} — MT5/FBS'.center(_W)}{'║'}")
    print(f"{'║'}{' EUR_USD Scalper | Kelly Criterion | XGBoost '.center(_W)}{'║'}")
    print(f"{'╚' + _line() + '╝'}\n")

def print_menu():
    items = [
        "[1] Demo  (cuenta demo FBS)",
        "[2] Real  (cuenta real FBS)  ← escribe CONFIRMO",
        "[3] Solo Backtest / Training",
        "[4] Análisis de logs",
        "[5] Optimizar parámetros",
        "[6] Salir",
    ]
    print(f"{'╔' + _line() + '╗'}")
    print(f"{'║'}{' Selecciona modo de operación:'.ljust(_W)}{'║'}")
    print(f"{'╠' + _line() + '╣'}")
    for item in items:
        print(f"{'║'}  {item.ljust(_W - 2)}{'║'}")
    print(f"{'╚' + _line() + '╝'}")

def print_table(headers: List[str], rows: List[List[Any]], col_width: int = 14):
    sep = "+" + "+".join("-" * col_width for _ in headers) + "+"
    print(sep)
    print("|" + "|".join(str(h).center(col_width) for h in headers) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(str(v).center(col_width) for v in row) + "|")
    print(sep)

def print_trade_open(symbol: str, direction: str, lots: float, entry: float,
                     sl: float, tp1: float, regime: str, prob: float):
    tag = "[>>>]"
    print(f"\n{tag} {symbol} {direction} {lots:.2f}lots @ {entry:.5f}")
    print(f"      SL:{sl:.5f} | TP1:{tp1:.5f} | {regime} | prob:{prob:.2%}")

def print_trade_close(symbol: str, direction: str, pips: float, pnl: float, reason: str):
    sign  = "WIN " if pnl >= 0 else "LOSS"
    arrow = "[+]" if pnl >= 0 else "[-]"
    print(f"\n{arrow} {sign} {symbol} {direction} | {pips:+.1f}pips | ${pnl:+.2f} | {reason}")

def update_dashboard(symbol: str, capital: float, equity: float,
                     pnl_day: float, trades_today: int, wins: int,
                     phase: int, pos_info: str = "Sin posicion"):
    wr = f"{wins/trades_today:.1%}" if trades_today > 0 else "N/A"
    line = (f"[{symbol}] Cap:${capital:.2f} Eq:${equity:.2f} "
            f"Dia:${pnl_day:+.2f} | {trades_today}t({wr}) | "
            f"Fase:{phase} | {pos_info}")
    sys.stdout.write(f"\r{line[:120]}")
    sys.stdout.flush()

def print_training_progress(iteration: int, max_iter: int, win_rate: float,
                             profit_factor: float, threshold: float):
    bar_len = 30
    filled  = int(bar_len * iteration / max_iter)
    bar     = "=" * filled + ">" + " " * (bar_len - filled - 1)
    status  = "OK " if win_rate >= 0.59 else "..."
    print(f"[{bar}] iter {iteration:2d}/{max_iter} | "
          f"WR:{win_rate:.1%} {status} | PF:{profit_factor:.2f} | thr:{threshold:.2f}")

def print_backtest_summary(metrics: dict):
    print(f"\n{'═' * _W}")
    print(f"  BACKTEST RESULTS")
    print(f"{'─' * _W}")
    rows = [
        ["Win Rate",        f"{metrics.get('win_rate', 0):.1%}"],
        ["Profit Factor",   f"{metrics.get('profit_factor', 0):.2f}"],
        ["Sharpe Ratio",    f"{metrics.get('sharpe', 0):.2f}"],
        ["Max Drawdown",    f"{metrics.get('max_drawdown', 0):.1%}"],
        ["Total Trades",    str(metrics.get('total_trades', 0))],
        ["Net PnL",         f"${metrics.get('net_pnl', 0):+.2f}"],
        ["Capital Final",   f"${metrics.get('final_capital', 0):.2f}"],
        ["ROI",             f"{metrics.get('roi', 0):.1%}"],
    ]
    for label, val in rows:
        print(f"  {label:<20} {val}")
    print(f"{'═' * _W}\n")

def print_equity_chart(equity_series: list, width: int = 54, height: int = 12):
    if len(equity_series) < 2:
        return
    mn, mx = min(equity_series), max(equity_series)
    if mx == mn:
        return
    print(f"\n  Equity Curve  (${mn:.2f} → ${mx:.2f})")
    chart = [[" "] * width for _ in range(height)]
    step  = max(1, len(equity_series) // width)
    for col, idx in enumerate(range(0, len(equity_series), step)):
        if col >= width:
            break
        val = equity_series[idx]
        row = height - 1 - int((val - mn) / (mx - mn) * (height - 1))
        row = max(0, min(height - 1, row))
        chart[row][col] = "*"
    for row in chart:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width + "+\n")
