import pandas as pd
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from config import settings

_log = get_logger("log_analyzer")


class LogAnalyzer:

    def load_all_trades(self, symbol: str = None) -> pd.DataFrame:
        dfs = []
        for f in sorted(settings.LOGS_TRADES_DIR.glob("trades_*.csv")):
            try:
                df = pd.read_csv(f)
                if symbol and "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                dfs.append(df)
            except Exception:
                continue
        # También carga de backtest
        for f in sorted(settings.LOGS_BACKTEST_DIR.glob("backtest_*.csv")):
            try:
                df = pd.read_csv(f)
                if symbol and "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                dfs.append(df)
            except Exception:
                continue
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def analyze(self, symbol: str = None) -> dict:
        df = self.load_all_trades(symbol)
        if df.empty:
            return {}

        result = {
            "by_hour":      self.by_hour(df),
            "by_direction": self.by_direction(df),
            "best_hours":   self.find_best_hours(df),
            "summary":      self._summary(df),
        }
        self._save(result, symbol)
        return result

    def by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        if "open_time" not in df.columns:
            return pd.DataFrame()
        try:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["open_time"], errors="coerce").dt.hour
            return df.groupby("hour")["pnl_usd"].agg(
                trades="count",
                win_rate=lambda x: (x > 0).mean(),
                pnl="sum",
                avg_pnl="mean",
            ).reset_index()
        except Exception:
            return pd.DataFrame()

    def by_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        if "direction" not in df.columns:
            return pd.DataFrame()
        return df.groupby("direction")["pnl_usd"].agg(
            trades="count",
            win_rate=lambda x: (x > 0).mean(),
            total_pnl="sum",
        ).reset_index()

    def find_best_hours(self, df: pd.DataFrame, top_n: int = 5) -> list:
        hourly = self.by_hour(df)
        if hourly.empty:
            return []
        good = hourly[hourly["trades"] >= 5].sort_values("win_rate", ascending=False)
        return good.head(top_n)["hour"].tolist()

    def find_worst_hours(self, df: pd.DataFrame, threshold: float = 0.40) -> list:
        hourly = self.by_hour(df)
        if hourly.empty:
            return []
        bad = hourly[(hourly["trades"] >= 5) & (hourly["win_rate"] < threshold)]
        return bad["hour"].tolist()

    def _summary(self, df: pd.DataFrame) -> dict:
        if df.empty or "pnl_usd" not in df.columns:
            return {}
        wins   = df[df["pnl_usd"] > 0]
        losses = df[df["pnl_usd"] <= 0]
        return {
            "total_trades":  len(df),
            "win_rate":      len(wins) / len(df),
            "profit_factor": wins["pnl_usd"].sum() / (losses["pnl_usd"].abs().sum() + 1e-9),
            "net_pnl":       df["pnl_usd"].sum(),
        }

    def _save(self, result: dict, symbol: str = None):
        path = settings.LOGS_SYSTEM_DIR / \
               f"analysis_{symbol or 'all'}_{datetime.utcnow().strftime('%Y%m%d')}.txt"
        settings.LOGS_SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
        lines = [f"Log Analysis — {symbol or 'ALL'} — {datetime.utcnow().isoformat()}"]
        if result.get("summary"):
            s = result["summary"]
            lines += [
                f"Total trades: {s.get('total_trades', 0)}",
                f"Win rate:     {s.get('win_rate', 0):.1%}",
                f"PF:           {s.get('profit_factor', 0):.2f}",
                f"Net PnL:      ${s.get('net_pnl', 0):+.2f}",
            ]
        if result.get("best_hours"):
            lines.append(f"Best hours UTC: {result['best_hours']}")
        path.write_text("\n".join(lines))
