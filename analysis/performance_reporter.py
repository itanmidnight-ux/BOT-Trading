import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger
from utils import display
from config import settings

_log = get_logger("performance_reporter")


class PerformanceReporter:

    def show(self, metrics: dict, symbol: str = ""):
        """Muestra reporte completo en terminal y guarda en archivo."""
        display.print_backtest_summary(metrics)

        if metrics.get("equity_curve"):
            display.print_equity_chart(metrics["equity_curve"])

        if metrics.get("trades_df") is not None:
            self._show_by_hour(metrics["trades_df"])

        self._save_report(metrics, symbol)

    def _show_by_hour(self, df: pd.DataFrame):
        if "open_time" not in df.columns or len(df) < 10:
            return
        try:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["open_time"], errors="coerce").dt.hour
            hourly = df.groupby("hour")["pnl_usd"].agg(
                trades="count",
                wr=lambda x: (x > 0).mean(),
                pnl="sum"
            ).reset_index()

            print("\n  Rendimiento por hora UTC:")
            print("  " + "-" * 40)
            print(f"  {'Hora':>4} | {'Trades':>6} | {'WR':>6} | {'PnL':>8}")
            print("  " + "-" * 40)
            for _, row in hourly.sort_values("wr", ascending=False).iterrows():
                print(f"  {int(row['hour']):>4}h | {int(row['trades']):>6} | "
                      f"{row['wr']:>5.1%} | ${row['pnl']:>+7.2f}")
        except Exception as e:
            _log.debug(f"Error en análisis por hora: {e}")

    def _save_report(self, metrics: dict, symbol: str):
        path = settings.LOGS_BACKTEST_DIR / \
               f"report_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt"
        settings.LOGS_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            f"BOT-Trading Report — {symbol} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 54,
            f"Win Rate:      {metrics.get('win_rate', 0):.1%}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Sharpe Ratio:  {metrics.get('sharpe', 0):.2f}",
            f"Max Drawdown:  {metrics.get('max_drawdown', 0):.1%}",
            f"Total Trades:  {metrics.get('total_trades', 0)}",
            f"Net PnL:       ${metrics.get('net_pnl', 0):+.2f}",
            f"Final Capital: ${metrics.get('final_capital', 0):.2f}",
            f"ROI:           {metrics.get('roi', 0):.1%}",
        ]
        path.write_text("\n".join(lines))
        _log.info(f"Reporte guardado: {path.name}")
