"""
data_validator.py
Valida la calidad de DataFrames de velas OHLCV descargados desde MT5.

Checks:
  - Gaps temporales anómalos (> umbral según TF)
  - Velas nulas / cero (errores de feed)
  - Outliers de precio (ventana rolling 200, ±5σ)
  - Consistencia OHLC: low ≤ min(open,close) ≤ max(open,close) ≤ high

Reparación:
  - forward-fill para gaps pequeños (< 3 velas consecutivas)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("data_validator")

# ── Umbrales de gaps por timeframe ────────────────────────────────────────────
# Un gap mayor que N períodos consecutivos se cuenta como gap anómalo
GAP_THRESHOLD: dict[str, int] = {
    "M1":  5,
    "M5":  3,
    "M15": 3,
    "H1":  3,
    "H4":  3,
}

# Segundos por período
TF_SECONDS: dict[str, int] = {
    "M1":  60,
    "M5":  300,
    "M15": 900,
    "H1":  3600,
    "H4":  14400,
}

OUTLIER_WINDOW = 200      # ventana rolling para media/std
OUTLIER_SIGMA  = 5.0      # desviaciones estándar para outlier
FWD_FILL_MAX   = 2        # máximo de velas consecutivas a rellenar (< 3)


# ── Resultado de validación ───────────────────────────────────────────────────

@dataclass
class ValidationReport:
    symbol: str
    timeframe_str: str
    n_total: int
    pct_valid: float          # 0.0–100.0
    n_gaps: int               # número de gaps anómalos detectados
    n_outliers: int           # velas con precio outlier
    n_zero_candles: int       # velas con cualquier campo OHLCV = 0
    n_ohlc_errors: int        # violaciones de low≤open/close≤high
    is_acceptable: bool       # True si pct_valid >= 95 %
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "OK" if self.is_acceptable else "RECHAZADO"
        return (
            f"[{self.symbol} {self.timeframe_str}] {status} | "
            f"válidas={self.pct_valid:.1f}% | "
            f"gaps={self.n_gaps} | outliers={self.n_outliers} | "
            f"ceros={self.n_zero_candles} | ohlc_err={self.n_ohlc_errors}"
        )


# ── Validador principal ───────────────────────────────────────────────────────

class DataValidator:
    """
    Valida y opcionalmente repara DataFrames de velas OHLCV.

    Uso típico:
        validator = DataValidator()
        report = validator.validate(df, "EURUSD", "M1")
        if report.is_acceptable:
            clean_df = validator.fix(df)
    """

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe_str: str,
    ) -> ValidationReport:
        """
        Ejecuta todos los checks sobre *df* y retorna un ValidationReport.

        Parameters
        ----------
        df            : DataFrame con columnas CANDLE_COLS y 'time' como datetime UTC.
        symbol        : Nombre del instrumento (solo para logging/reporte).
        timeframe_str : String del TF ("M1", "M5", etc.).
        """
        issues: list[str] = []
        n_total = len(df)

        if n_total == 0:
            log.warning("[%s %s] DataFrame vacío.", symbol, timeframe_str)
            return ValidationReport(
                symbol=symbol,
                timeframe_str=timeframe_str,
                n_total=0,
                pct_valid=0.0,
                n_gaps=0,
                n_outliers=0,
                n_zero_candles=0,
                n_ohlc_errors=0,
                is_acceptable=False,
                issues=["DataFrame vacío"],
            )

        # ── 1. Gaps temporales ─────────────────────────────────────────────
        n_gaps = self._check_gaps(df, symbol, timeframe_str, issues)

        # ── 2. Velas cero ──────────────────────────────────────────────────
        n_zero = self._check_zero_candles(df, symbol, timeframe_str, issues)

        # ── 3. Outliers de precio ──────────────────────────────────────────
        n_outliers = self._check_outliers(df, symbol, timeframe_str, issues)

        # ── 4. Consistencia OHLC ───────────────────────────────────────────
        n_ohlc_err = self._check_ohlc_consistency(df, symbol, timeframe_str, issues)

        # ── Calcular pct_valid ─────────────────────────────────────────────
        # Filas problemáticas: unión de ceros, outliers y errores OHLC
        mask_zero     = self._zero_mask(df)
        mask_outlier  = self._outlier_mask(df)
        mask_ohlc     = self._ohlc_error_mask(df)
        bad_rows = (mask_zero | mask_outlier | mask_ohlc).sum()
        pct_valid = max(0.0, (n_total - bad_rows) / n_total * 100.0)

        is_acceptable = pct_valid >= 95.0

        report = ValidationReport(
            symbol=symbol,
            timeframe_str=timeframe_str,
            n_total=n_total,
            pct_valid=round(pct_valid, 4),
            n_gaps=n_gaps,
            n_outliers=n_outliers,
            n_zero_candles=n_zero,
            n_ohlc_errors=n_ohlc_err,
            is_acceptable=is_acceptable,
            issues=issues,
        )

        log.info(report.summary())
        if issues:
            for issue in issues:
                log.debug("  issue: %s", issue)

        return report

    # ── Fix / reparación ──────────────────────────────────────────────────

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica forward-fill para gaps pequeños (< 3 velas consecutivas).

        Estrategia:
        - Detecta posiciones donde falta al menos 1 período.
        - Si el gap tiene menos de FWD_FILL_MAX+1 períodos faltantes se interpolan
          las filas insertando copias de la vela anterior (forward-fill).
        - Gaps mayores o iguales a FWD_FILL_MAX+1 no se tocan.

        Retorna un nuevo DataFrame sin modificar el original.
        """
        if df.empty:
            return df.copy()

        df = df.copy().sort_values("time").reset_index(drop=True)

        # Determinar el período predominante
        diffs = df["time"].diff().dt.total_seconds().dropna()
        if diffs.empty:
            return df

        period_s = diffs.mode().iloc[0]  # período más frecuente en segundos

        # Iterar en orden inverso para poder insertar filas sin afectar índices
        rows_to_insert: list[dict] = []
        for i in range(1, len(df)):
            gap_s = (df.loc[i, "time"] - df.loc[i - 1, "time"]).total_seconds()
            missing = int(round(gap_s / period_s)) - 1  # velas que faltan
            if 0 < missing <= FWD_FILL_MAX:
                prev_row = df.loc[i - 1].copy()
                for step in range(1, missing + 1):
                    new_row = prev_row.copy()
                    new_row["time"] = prev_row["time"] + pd.Timedelta(seconds=period_s * step)
                    rows_to_insert.append(new_row.to_dict())

        if rows_to_insert:
            insert_df = pd.DataFrame(rows_to_insert)
            df = pd.concat([df, insert_df], ignore_index=True)
            df = df.sort_values("time").reset_index(drop=True)
            log.info("fix(): %d filas insertadas por forward-fill.", len(rows_to_insert))
        else:
            log.debug("fix(): no se requirió forward-fill.")

        return df

    # ── Métodos privados de detección ─────────────────────────────────────

    def _check_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        tf: str,
        issues: list[str],
    ) -> int:
        """Cuenta gaps temporales superiores al umbral configurado para el TF."""
        threshold_periods = GAP_THRESHOLD.get(tf, 3)
        period_s = TF_SECONDS.get(tf, 60)
        threshold_s = threshold_periods * period_s

        diffs = df["time"].sort_values().diff().dt.total_seconds().dropna()
        large_gaps = diffs[diffs > threshold_s]
        n_gaps = len(large_gaps)

        if n_gaps > 0:
            max_gap_s = large_gaps.max()
            msg = (
                f"[{symbol} {tf}] {n_gaps} gap(s) temporal(es) detectado(s). "
                f"Mayor gap: {max_gap_s:.0f}s ({max_gap_s/period_s:.1f} períodos)."
            )
            issues.append(msg)
            log.warning(msg)

        return n_gaps

    @staticmethod
    def _zero_mask(df: pd.DataFrame) -> pd.Series:
        """Máscara booleana de filas con open, high, low, close o tick_volume = 0."""
        cols = [c for c in ["open", "high", "low", "close", "tick_volume"] if c in df.columns]
        mask = (df[cols] == 0).any(axis=1)
        return mask

    def _check_zero_candles(
        self,
        df: pd.DataFrame,
        symbol: str,
        tf: str,
        issues: list[str],
    ) -> int:
        mask = self._zero_mask(df)
        n_zero = int(mask.sum())
        if n_zero > 0:
            msg = f"[{symbol} {tf}] {n_zero} vela(s) con valor OHLCV = 0 (errores de feed)."
            issues.append(msg)
            log.warning(msg)
        return n_zero

    @staticmethod
    def _outlier_mask(df: pd.DataFrame) -> pd.Series:
        """
        Máscara booleana de filas donde 'close' supera mean ± OUTLIER_SIGMA*std
        en una ventana rolling de OUTLIER_WINDOW velas.
        """
        if "close" not in df.columns or len(df) < OUTLIER_WINDOW // 2:
            return pd.Series(False, index=df.index)

        close = df["close"]
        roll_mean = close.rolling(OUTLIER_WINDOW, min_periods=20, center=True).mean()
        roll_std  = close.rolling(OUTLIER_WINDOW, min_periods=20, center=True).std()

        upper = roll_mean + OUTLIER_SIGMA * roll_std
        lower = roll_mean - OUTLIER_SIGMA * roll_std

        mask = (close > upper) | (close < lower)
        return mask.fillna(False)

    def _check_outliers(
        self,
        df: pd.DataFrame,
        symbol: str,
        tf: str,
        issues: list[str],
    ) -> int:
        mask = self._outlier_mask(df)
        n_out = int(mask.sum())
        if n_out > 0:
            msg = (
                f"[{symbol} {tf}] {n_out} vela(s) con precio outlier "
                f"(> {OUTLIER_SIGMA}σ en ventana {OUTLIER_WINDOW})."
            )
            issues.append(msg)
            log.warning(msg)
        return n_out

    @staticmethod
    def _ohlc_error_mask(df: pd.DataFrame) -> pd.Series:
        """
        Máscara de filas que violan: low ≤ min(open,close) ≤ max(open,close) ≤ high
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return pd.Series(False, index=df.index)

        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]

        mask = (
            (l > o) | (l > c) |        # low > open o low > close
            (h < o) | (h < c) |        # high < open o high < close
            (h < l)                     # high < low (imposible)
        )
        return mask.fillna(False)

    def _check_ohlc_consistency(
        self,
        df: pd.DataFrame,
        symbol: str,
        tf: str,
        issues: list[str],
    ) -> int:
        mask = self._ohlc_error_mask(df)
        n_err = int(mask.sum())
        if n_err > 0:
            msg = (
                f"[{symbol} {tf}] {n_err} vela(s) con violación OHLC "
                "(low > open/close o high < open/close)."
            )
            issues.append(msg)
            log.warning(msg)
        return n_err
