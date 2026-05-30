"""
ModelUpdater — gestiona el ciclo de vida de los modelos XGBoost en producción:
carga, re-entrenamiento condicional, comparación y rollback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb

from config import settings
from core.model_trainer import ModelTrainer
from core.model_evaluator import ModelEvaluator
from utils.logger import get_logger

logger = get_logger("model_updater")


class ModelUpdater:
    """Mantiene en memoria el modelo activo y orquesta re-entrenamientos."""

    def __init__(self, symbol: str) -> None:
        self._models: dict[str, xgb.XGBClassifier | None] = {}
        self._trainer  = ModelTrainer()
        self._evaluator = ModelEvaluator()

        # Pre-cargar el modelo más reciente si existe
        model = self.get_model(symbol)
        if model is not None:
            logger.info("[%s] Modelo cargado desde checkpoint latest.", symbol)
        else:
            logger.info("[%s] No se encontró checkpoint previo.", symbol)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def get_model(self, symbol: str) -> xgb.XGBClassifier | None:
        """Carga (o devuelve desde caché) el modelo latest del símbolo.

        Returns
        -------
        XGBClassifier | None
            El modelo si existe el symlink ``{symbol}_latest.pkl``, o ``None``.
        """
        if symbol in self._models and self._models[symbol] is not None:
            return self._models[symbol]

        symlink = self._symlink_path(symbol)
        if not symlink.exists() and not symlink.is_symlink():
            self._models[symbol] = None
            return None

        try:
            model = joblib.load(symlink)
            self._models[symbol] = model
            logger.debug("[%s] Modelo cargado desde %s.", symbol, symlink)
            return model
        except Exception as exc:
            logger.error("[%s] Error cargando modelo: %s", symbol, exc)
            self._models[symbol] = None
            return None

    # ------------------------------------------------------------------

    def maybe_retrain(
        self,
        symbol: str,
        df_features: pd.DataFrame,
        data_updater: Any,
    ) -> bool:
        """Re-entrena si ``data_updater.should_retrain(symbol)`` lo indica.

        Parameters
        ----------
        symbol:
            Identificador del instrumento.
        df_features:
            DataFrame con features + columna ``target``.
        data_updater:
            Objeto con método ``should_retrain(symbol) -> bool``.

        Returns
        -------
        bool
            ``True`` si se realizó y aceptó un nuevo entrenamiento.
        """
        if not data_updater.should_retrain(symbol):
            logger.debug("[%s] Re-entrenamiento no requerido.", symbol)
            return False

        logger.info("[%s] Iniciando re-entrenamiento...", symbol)

        # Métricas del modelo anterior (puede ser None si no hay reporte)
        old_metrics = self._evaluator.load_latest_metrics(symbol)

        # Entrenar nuevo modelo
        try:
            new_metrics = self._trainer.train(df_features, symbol)
        except Exception as exc:
            logger.error("[%s] Error durante re-entrenamiento: %s", symbol, exc)
            return False

        # Si no hay modelo anterior, el nuevo siempre gana
        if old_metrics is None:
            logger.info("[%s] Sin modelo previo → aceptando nuevo modelo.", symbol)
            self._models[symbol] = None  # Forzar recarga desde symlink actualizado
            self.get_model(symbol)
            return True

        # Comparar métricas
        self._evaluator.print_comparison(new_metrics, old_metrics)

        if self._evaluator.is_better_than(new_metrics, old_metrics):
            logger.info(
                "[%s] Nuevo modelo es MEJOR → activando. f1=%.4f→%.4f, wr=%.4f→%.4f",
                symbol,
                old_metrics.get("f1", 0.0),    new_metrics["f1"],
                old_metrics.get("win_rate", 0.0), new_metrics["win_rate"],
            )
            # El symlink ya fue actualizado por ModelTrainer._save_checkpoint()
            # Recargar modelo en memoria
            self._models[symbol] = None
            self.get_model(symbol)
            return True
        else:
            logger.warning(
                "[%s] Nuevo modelo es PEOR o IGUAL → manteniendo anterior. "
                "f1: %.4f vs %.4f | wr: %.4f vs %.4f",
                symbol,
                new_metrics["f1"],               old_metrics.get("f1", 0.0),
                new_metrics.get("win_rate", 0.0), old_metrics.get("win_rate", 0.0),
            )
            # Revertir el symlink al checkpoint anterior
            self._restore_symlink_to_previous(symbol)
            return False

    # ------------------------------------------------------------------

    def rollback(self, symbol: str) -> None:
        """Restaura el checkpoint anterior si existe.

        Actualiza el symlink ``{symbol}_latest.pkl`` para que apunte al
        penúltimo checkpoint encontrado en MODELS_DIR.
        """
        previous = self._get_previous_checkpoint(symbol)
        if previous is None:
            logger.warning("[%s] No hay checkpoint anterior al que hacer rollback.", symbol)
            return

        symlink = self._symlink_path(symbol)
        _update_symlink(symlink, previous)
        logger.info("[%s] Rollback a: %s", symbol, previous.name)

        # Recargar en memoria
        self._models[symbol] = None
        self.get_model(symbol)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _get_previous_checkpoint(self, symbol: str) -> Path | None:
        """Retorna el penúltimo checkpoint del símbolo, o None si hay menos de 2.

        Los checkpoints siguen el patrón ``{symbol}_model_{YYYYMMDD_HHMMSS}.pkl``
        y se ordenan lexicográficamente (el timestamp en el nombre garantiza
        el orden cronológico).
        """
        models_dir = Path(settings.MODELS_DIR)
        if not models_dir.exists():
            return None

        checkpoints = sorted(
            models_dir.glob(f"{symbol}_model_*.pkl"),
        )

        if len(checkpoints) < 2:
            return None

        # El penúltimo (el último es el más reciente)
        return checkpoints[-2]

    def _symlink_path(self, symbol: str) -> Path:
        return Path(settings.MODELS_DIR) / f"{symbol}_latest.pkl"

    def _restore_symlink_to_previous(self, symbol: str) -> None:
        """Tras un re-entrenamiento fallido, revierte el symlink al checkpoint anterior."""
        previous = self._get_previous_checkpoint(symbol)
        if previous is None:
            logger.debug("[%s] Sin checkpoint anterior; no se revierte symlink.", symbol)
            return

        symlink = self._symlink_path(symbol)
        _update_symlink(symlink, previous)
        logger.info(
            "[%s] Symlink revertido a checkpoint anterior: %s",
            symbol, previous.name,
        )

        # Recargar modelo en memoria desde el checkpoint anterior
        self._models[symbol] = None
        self.get_model(symbol)


# ------------------------------------------------------------------
# Helper de symlinks
# ------------------------------------------------------------------

def _update_symlink(symlink: Path, target: Path) -> None:
    """Actualiza (o crea) un symlink atómicamente."""
    symlink.parent.mkdir(parents=True, exist_ok=True)
    if symlink.exists() or symlink.is_symlink():
        symlink.unlink()
    os.symlink(target.resolve(), symlink)
