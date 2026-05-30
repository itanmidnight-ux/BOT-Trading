"""
ModelEvaluator — evalúa un modelo XGBoost ya entrenado sobre un test set,
compara métricas con el modelo anterior y carga/muestra reportes históricos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import settings
from utils.logger import get_logger

logger = get_logger("model_evaluator")


class ModelEvaluator:
    """Calcula métricas completas de un modelo y facilita comparaciones."""

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_test: pd.DataFrame,
        symbol: str,
    ) -> dict[str, Any]:
        """Evalúa ``model`` sobre el test set y retorna métricas completas.

        Parameters
        ----------
        model:
            Clasificador XGBoost ya entrenado.
        X_test:
            Array 2-D de features del test set (shape [n, n_features]).
        y_test:
            Array 1-D con etiquetas binarias (0/1) del test set.
        df_test:
            DataFrame original del test set (necesario para la columna ``close``
            y los nombres de features del modelo).
        symbol:
            Identificador del instrumento.

        Returns
        -------
        dict
            accuracy, precision, recall, f1, auc_roc, win_rate,
            profit_factor, feature_importance_top10.
        """
        probas = model.predict_proba(X_test)[:, 1]
        y_pred = (probas >= settings.SIGNAL_THRESHOLD).astype(int)

        acc  = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec  = float(recall_score(y_test, y_pred, zero_division=0))
        f1   = float(f1_score(y_test, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_test, probas))
        except ValueError:
            auc = 0.0

        win_rate      = self._calc_win_rate(probas, df_test)
        profit_factor = self._calc_profit_factor(probas, df_test)
        top10         = self._feature_importance_top10(model, df_test)

        metrics: dict[str, Any] = {
            "symbol":                   symbol,
            "accuracy":                 acc,
            "precision":                prec,
            "recall":                   rec,
            "f1":                       f1,
            "auc_roc":                  auc,
            "win_rate":                 win_rate,
            "profit_factor":            profit_factor,
            "feature_importance_top10": top10,
        }

        logger.info(
            "[%s] Evaluación → f1=%.4f | wr=%.4f | pf=%.4f | auc=%.4f",
            symbol, f1, win_rate, profit_factor, auc,
        )
        return metrics

    # ------------------------------------------------------------------

    def is_better_than(
        self,
        new_metrics: dict[str, Any],
        old_metrics: dict[str, Any],
    ) -> bool:
        """True si el nuevo modelo mejora (o iguala) al anterior en f1 Y win_rate.

        Parameters
        ----------
        new_metrics:
            Métricas del modelo candidato.
        old_metrics:
            Métricas del modelo actualmente en producción.

        Returns
        -------
        bool
            ``True`` si ``new_metrics["f1"] >= old_metrics["f1"]``
            **y** ``new_metrics["win_rate"] >= old_metrics["win_rate"]``.
        """
        new_f1  = float(new_metrics.get("f1",       0.0))
        old_f1  = float(old_metrics.get("f1",       0.0))
        new_wr  = float(new_metrics.get("win_rate", 0.0))
        old_wr  = float(old_metrics.get("win_rate", 0.0))

        result = (new_f1 >= old_f1) and (new_wr >= old_wr)
        logger.debug(
            "is_better_than → f1: %.4f vs %.4f | wr: %.4f vs %.4f → %s",
            new_f1, old_f1, new_wr, old_wr, result,
        )
        return result

    # ------------------------------------------------------------------

    def load_latest_metrics(self, symbol: str) -> dict[str, Any] | None:
        """Carga el reporte JSON más reciente del símbolo desde REPORTS_DIR.

        Returns
        -------
        dict | None
            Diccionario de métricas, o ``None`` si no existe ningún reporte.
        """
        reports_dir = Path(settings.REPORTS_DIR)
        if not reports_dir.exists():
            return None

        # Busca todos los reportes del símbolo y toma el más reciente
        pattern = f"{symbol}_report_*.json"
        reports = sorted(reports_dir.glob(pattern))
        if not reports:
            logger.info("[%s] No se encontró reporte previo en %s.", symbol, reports_dir)
            return None

        latest = reports[-1]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
            logger.info("[%s] Reporte cargado: %s", symbol, latest.name)
            return data
        except Exception as exc:
            logger.error("[%s] Error leyendo reporte %s: %s", symbol, latest, exc)
            return None

    # ------------------------------------------------------------------

    def print_comparison(
        self,
        new_m: dict[str, Any],
        old_m: dict[str, Any],
    ) -> None:
        """Imprime una tabla comparativa en terminal entre modelo nuevo y anterior."""
        keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "win_rate", "profit_factor"]

        col_w   = 12
        label_w = 18

        header = f"{'Métrica':<{label_w}} {'Nuevo':>{col_w}} {'Anterior':>{col_w}} {'Delta':>{col_w}}"
        sep    = "-" * len(header)

        print(sep)
        print("  Comparación de modelos")
        print(sep)
        print(header)
        print(sep)

        for key in keys:
            nv  = float(new_m.get(key, 0.0))
            ov  = float(old_m.get(key, 0.0))
            delta = nv - ov
            sign  = "+" if delta >= 0 else ""
            print(
                f"{key:<{label_w}} {nv:>{col_w}.4f} {ov:>{col_w}.4f} "
                f"{sign}{delta:>{col_w - 1}.4f}"
            )

        print(sep)
        verdict = "MEJOR" if self.is_better_than(new_m, old_m) else "PEOR / IGUAL"
        print(f"  Veredicto: {verdict}")
        print(sep)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _calc_win_rate(
        self,
        probas: np.ndarray,
        df_test: pd.DataFrame,
    ) -> float:
        """Fracción de trades simulados que aciertan la dirección."""
        if "close" not in df_test.columns:
            logger.warning("df_test sin columna 'close'; win_rate=0.")
            return 0.0

        closes    = df_test["close"].values
        threshold = float(settings.SIGNAL_THRESHOLD)
        n         = len(closes) - 1

        if n <= 0:
            return 0.0

        wins = trades = 0
        for i in range(n):
            p = probas[i]
            if p > threshold:
                trades += 1
                if closes[i + 1] > closes[i]:
                    wins += 1
            elif p < (1.0 - threshold):
                trades += 1
                if closes[i + 1] < closes[i]:
                    wins += 1

        return float(wins / trades) if trades > 0 else 0.0

    def _calc_profit_factor(
        self,
        probas: np.ndarray,
        df_test: pd.DataFrame,
    ) -> float:
        """Profit Factor simulado en pips (suma ganancias / suma pérdidas absolutas).

        Se consideran únicamente velas con señal activa
        (proba > threshold para BUY, proba < 1-threshold para SELL).
        El P&L de cada trade es ``|close[+1] - close[0]|`` con signo según
        si acertó o erró la dirección.
        """
        if "close" not in df_test.columns:
            return 0.0

        closes    = df_test["close"].values
        threshold = float(settings.SIGNAL_THRESHOLD)
        n         = len(closes) - 1

        gains  = 0.0
        losses = 0.0

        for i in range(n):
            p       = probas[i]
            move    = closes[i + 1] - closes[i]

            if p > threshold:          # BUY
                pnl = move
            elif p < (1.0 - threshold):  # SELL
                pnl = -move
            else:
                continue

            if pnl > 0:
                gains  += pnl
            else:
                losses += abs(pnl)

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return float(gains / losses)

    def _feature_importance_top10(
        self,
        model: xgb.XGBClassifier,
        df_test: pd.DataFrame,
    ) -> list[tuple[str, float]]:
        """Retorna las 10 features más importantes según ``feature_importances_``."""
        try:
            importances = model.feature_importances_
        except AttributeError:
            return []

        # Intentar recuperar nombres de features desde el booster
        try:
            feature_names = model.get_booster().feature_names
            if feature_names is None:
                raise ValueError
        except Exception:
            # Fallback: usar nombres del df_test (sin columnas no-feature)
            _non = {"time", "open", "high", "low", "close", "tick_volume", "spread", "target"}
            feature_names = [c for c in df_test.columns if c not in _non]

        if len(feature_names) != len(importances):
            # No se puede mapear nombres con certeza
            feature_names = [f"f{i}" for i in range(len(importances))]

        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:10]
