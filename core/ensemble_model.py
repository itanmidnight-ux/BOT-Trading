import os
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import settings
from utils.logger import get_logger

logger = get_logger("ensemble_model")


class EnsembleModel:
    def __init__(self):
        self.models: dict = {}
        self.weights: dict = {}
        self._checkpoint_dir = Path(settings.MODELS_DIR)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) -> dict:
        xgb_params = {k: v for k, v in settings.XGB_PARAMS.items()}
        early_xgb = xgb_params.pop("early_stopping_rounds", 50)

        lgb_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": -1,
        }

        cat_params = {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.02,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": 0,
            "early_stopping_rounds": 40,
        }

        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=early_xgb)
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        logger.info("Training CatBoost...")
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )

        self.models = {"xgb": xgb_model, "lgb": lgb_model, "cat": cat_model}

        aucs = {
            name: roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            for name, model in self.models.items()
        }
        total = sum(aucs.values())
        self.weights = {k: v / total for k, v in aucs.items()}

        logger.info(
            "Validation AUCs — XGB: %.4f  LGB: %.4f  CAT: %.4f",
            aucs["xgb"],
            aucs["lgb"],
            aucs["cat"],
        )
        logger.info(
            "Ensemble weights — XGB: %.3f  LGB: %.3f  CAT: %.3f",
            self.weights["xgb"],
            self.weights["lgb"],
            self.weights["cat"],
        )

        test_proba = self.predict_proba(X_test)
        test_preds = (test_proba >= 0.5).astype(int)

        metrics = {
            "val_auc": {name: round(auc, 6) for name, auc in aucs.items()},
            "weights": {name: round(w, 6) for name, w in self.weights.items()},
            "test_auc": round(roc_auc_score(y_test, test_proba), 6),
            "test_accuracy": round(accuracy_score(y_test, test_preds), 6),
            "test_f1": round(f1_score(y_test, test_preds, zero_division=0), 6),
        }

        logger.info(
            "Test — AUC: %.4f  Acc: %.4f  F1: %.4f",
            metrics["test_auc"],
            metrics["test_accuracy"],
            metrics["test_f1"],
        )
        return metrics

    def predict_proba(self, X) -> np.ndarray:
        if not self.models:
            raise RuntimeError("EnsembleModel has no trained models. Call train() or load() first.")
        weighted = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 1.0 / len(self.models))
            weighted += w * model.predict_proba(X)[:, 1]
        return weighted

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("EnsembleModel has no trained models. Call train() or load() first.")

        probas = {
            name: model.predict_proba(X)[:, 1]
            for name, model in self.models.items()
        }
        weighted = sum(self.weights.get(n, 1.0 / len(self.models)) * p for n, p in probas.items())

        top_name = max(self.weights, key=self.weights.get)
        top_proba = probas[top_name]
        high_confidence = (np.abs(top_proba - weighted) <= 0.05).astype(float)

        return weighted, high_confidence

    def save(self, symbol: str) -> None:
        path = self._checkpoint_dir / f"{symbol}_ensemble.pkl"
        payload = {"models": self.models, "weights": self.weights}
        joblib.dump(payload, path)
        logger.info("EnsembleModel saved → %s", path)

    def load(self, symbol: str) -> None:
        path = self._checkpoint_dir / f"{symbol}_ensemble.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No ensemble checkpoint found at {path}")
        payload = joblib.load(path)
        self.models = payload["models"]
        self.weights = payload["weights"]
        logger.info(
            "EnsembleModel loaded from %s (weights: %s)",
            path,
            {k: round(v, 3) for k, v in self.weights.items()},
        )

    def get_weights(self) -> dict:
        return dict(self.weights)
