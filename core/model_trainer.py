"""
ModelTrainer — entrena un clasificador XGBoost sobre features precalculadas,
evalúa en test set, guarda checkpoint y reporte JSON.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
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

logger = get_logger("model_trainer")

# Columnas que nunca son features
_NON_FEATURE_COLS = {
    "time", "open", "high", "low", "close",
    "tick_volume", "spread", "target",
}


class ModelTrainer:
    """Entrena, evalúa y persiste modelos XGBoost de forma reproducible."""

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def train(self, df_features: pd.DataFrame, symbol: str) -> dict[str, Any]:
        """Entrena el modelo y retorna un diccionario con todas las métricas.

        Parameters
        ----------
        df_features:
            DataFrame con columnas de features **más** la columna ``target``
            (binaria: 0 o 1). Puede incluir columnas OHLCV que se descartan
            automáticamente.
        symbol:
            Identificador del instrumento (p. ej. ``"EURUSD"``).

        Returns
        -------
        dict
            Métricas del modelo entrenado: accuracy, precision, recall, f1,
            auc_roc, win_rate, n_train, n_val, n_test y rutas de artefactos.
        """
        logger.info("[%s] Iniciando entrenamiento. Filas totales: %d", symbol, len(df_features))

        df = df_features.copy()

        # --- 1. Limpiar NaN -----------------------------------------------
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            logger.warning("[%s] Se eliminaron %d filas con NaN.", symbol, dropped)

        if len(df) < 50:
            raise ValueError(
                f"[{symbol}] Datos insuficientes tras eliminar NaN: {len(df)} filas."
            )

        # --- 2. Separar features / target ----------------------------------
        feature_cols = self.get_feature_cols(df)
        if not feature_cols:
            raise ValueError(f"[{symbol}] No se encontraron columnas de features.")

        X = df[feature_cols].values
        y = df["target"].values

        # --- 3. Split temporal 70 / 15 / 15 --------------------------------
        n = len(df)
        train_end = int(n * settings.WALK_FORWARD_TRAIN)
        val_end   = int(n * (settings.WALK_FORWARD_TRAIN + settings.WALK_FORWARD_VAL))

        X_train, y_train = X[:train_end],         y[:train_end]
        X_val,   y_val   = X[train_end:val_end],  y[train_end:val_end]
        X_test,  y_test  = X[val_end:],           y[val_end:]
        df_test          = df.iloc[val_end:].reset_index(drop=True)

        logger.info(
            "[%s] Split → train=%d | val=%d | test=%d",
            symbol, len(X_train), len(X_val), len(X_test),
        )

        # --- 4. Balanceo de clases ------------------------------------------
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        params = dict(settings.XGB_PARAMS)  # copia para no mutar el original

        if n_pos == 0 or n_neg == 0:
            logger.warning("[%s] Clase minoritaria ausente en train; sin balanceo.", symbol)
        else:
            ratio = max(n_pos, n_neg) / min(n_pos, n_neg)
            if ratio > (60 / 40):  # desbalance > 60/40
                scale = n_neg / n_pos
                params["scale_pos_weight"] = scale
                logger.info("[%s] scale_pos_weight=%.3f (n_neg=%d, n_pos=%d)", symbol, scale, n_neg, n_pos)

        # --- 5. Entrenamiento -----------------------------------------------
        # early_stopping_rounds puede venir en XGB_PARAMS; si no, lo forzamos
        params.setdefault("early_stopping_rounds", 50)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = getattr(model, "best_iteration", params.get("n_estimators", 600))
        logger.info("[%s] Mejor iteración (early stopping): %d", symbol, best_iter)

        # --- 6. Métricas en test set ----------------------------------------
        metrics = self._evaluate_metrics(model, X_test, y_test, df_test, feature_cols)
        metrics.update({
            "symbol":  symbol,
            "n_train": int(len(X_train)),
            "n_val":   int(len(X_val)),
            "n_test":  int(len(X_test)),
            "best_iteration": int(best_iter),
            "feature_cols":   feature_cols,
        })

        logger.info(
            "[%s] Test → acc=%.4f | prec=%.4f | rec=%.4f | f1=%.4f | auc=%.4f | wr=%.4f",
            symbol,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"],
            metrics["auc_roc"],  metrics["win_rate"],
        )

        # --- 7. Persistencia ------------------------------------------------
        checkpoint_path = self._save_checkpoint(model, symbol)
        report_path     = self._save_report(metrics, symbol)

        metrics["checkpoint_path"] = str(checkpoint_path)
        metrics["report_path"]     = str(report_path)

        return metrics

    # ------------------------------------------------------------------

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Retorna las columnas que son features (excluye OHLCV, target, time)."""
        return [c for c in df.columns if c not in _NON_FEATURE_COLS]

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _evaluate_metrics(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_test: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """Calcula métricas estándar + win_rate simulado."""
        probas   = model.predict_proba(X_test)[:, 1]
        y_pred   = (probas >= settings.SIGNAL_THRESHOLD).astype(int)

        acc  = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec  = float(recall_score(y_test, y_pred, zero_division=0))
        f1   = float(f1_score(y_test, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_test, probas))
        except ValueError:
            auc = 0.0

        win_rate = self._simulate_win_rate(probas, df_test)

        return {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc_roc":   auc,
            "win_rate":  win_rate,
        }

    def _simulate_win_rate(
        self,
        probas: np.ndarray,
        df_test: pd.DataFrame,
    ) -> float:
        """Win rate simulado con threshold 0.62.

        BUY  si proba > 0.62  → win si close[+1] > close[0]
        SELL si proba <= 0.62 → win si close[+1] < close[0]
        Se ignoran la última fila (no hay close[+1]).
        """
        if "close" not in df_test.columns:
            logger.warning("df_test sin columna 'close'; win_rate=0.")
            return 0.0

        threshold = 0.62
        closes    = df_test["close"].values
        n         = len(closes) - 1  # última fila sin siguiente vela

        if n <= 0:
            return 0.0

        wins = 0
        trades = 0

        for i in range(n):
            p = probas[i]
            is_buy  = p > threshold
            is_sell = p < (1.0 - threshold)

            if not (is_buy or is_sell):
                continue  # zona neutral → no operar

            trades += 1
            if is_buy and closes[i + 1] > closes[i]:
                wins += 1
            elif is_sell and closes[i + 1] < closes[i]:
                wins += 1

        if trades == 0:
            return 0.0

        return float(wins / trades)

    # ------------------------------------------------------------------

    def _save_checkpoint(self, model: xgb.XGBClassifier, symbol: str) -> Path:
        """Guarda el modelo y actualiza el symlink ``{symbol}_latest.pkl``."""
        models_dir = Path(settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{symbol}_model_{timestamp}.pkl"
        checkpoint_path = models_dir / checkpoint_name

        joblib.dump(model, checkpoint_path)
        logger.info("[%s] Checkpoint guardado: %s", symbol, checkpoint_path)

        # Actualiza (o crea) el symlink apuntando al checkpoint nuevo
        symlink = models_dir / f"{symbol}_latest.pkl"
        if symlink.exists() or symlink.is_symlink():
            symlink.unlink()
        os.symlink(checkpoint_path.resolve(), symlink)
        logger.info("[%s] Symlink actualizado: %s → %s", symbol, symlink.name, checkpoint_name)

        return checkpoint_path

    def _save_report(self, metrics: dict[str, Any], symbol: str) -> Path:
        """Guarda reporte JSON en REPORTS_DIR."""
        reports_dir = Path(settings.REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)

        date_str     = datetime.now().strftime("%Y%m%d")
        report_name  = f"{symbol}_report_{date_str}.json"
        report_path  = reports_dir / report_name

        # Serializar: convertir tipos numpy a Python nativos
        serializable = _make_serializable(metrics)

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, ensure_ascii=False)

        logger.info("[%s] Reporte guardado: %s", symbol, report_path)
        return report_path


# ------------------------------------------------------------------
# Helpers de serialización
# ------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Convierte recursivamente tipos numpy/Path a tipos Python nativos."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
