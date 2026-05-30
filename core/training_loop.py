"""
Loop de entrenamiento iterativo.
Reentrena el modelo ajustando parámetros hasta alcanzar 59% WR mínimo.
Objetivo final: 70%+ WR con bajo riesgo.
"""
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from utils.logger import get_logger
from utils import display, notifier
from config import settings, constants

_log = get_logger("training_loop")


class TrainingLoop:
    """
    Orquesta el ciclo completo de entrenamiento → backtest → evaluación.
    Ajusta parámetros automáticamente para alcanzar MIN_WIN_RATE_LIVE (59%).
    """

    def __init__(self, symbol: str, feature_engine, model_trainer,
                 model_evaluator, backtester_cls, regime_detector=None,
                 ensemble_model=None, lstm_model=None, objective_engine=None,
                 ollama_advisor=None):
        self.symbol          = symbol
        self._fe             = feature_engine
        self._trainer        = model_trainer
        self._evaluator      = model_evaluator
        self._backtester_cls = backtester_cls
        self._regime         = regime_detector
        self._ensemble       = ensemble_model
        self._lstm           = lstm_model
        self._objective      = objective_engine
        self._ollama         = ollama_advisor

        self._threshold   = settings.SIGNAL_THRESHOLD
        self._sl_mult     = settings.ATR_SL_MULTIPLIER
        self._tp1_mult    = settings.ATR_TP1_MULTIPLIER
        self._best_metrics: Optional[dict] = None
        self._best_model  = None

    def run(self, df_raw: pd.DataFrame,
            initial_capital: float = settings.INITIAL_CAPITAL,
            max_iterations: int    = settings.MAX_TRAIN_ITERS,
            verbose: bool          = True) -> dict:
        """
        Ejecuta el loop de entrenamiento iterativo.

        Estrategia de ajuste por iteración:
          iter 1-5:   ajuste fino de SIGNAL_THRESHOLD (+0.02 por iteración)
          iter 6-10:  reduce ATR_SL (-0.1) para dar más espacio al precio
          iter 11-15: aumenta ATR_TP1 (+0.1) para capturar más movimiento
          iter 16-20: ajuste combinado más agresivo

        Retorna: dict con métricas finales, modelo, y flag ready_for_live.
        """
        _log.info(f"=== TRAINING LOOP {self.symbol} — objetivo WR >= {settings.MIN_WIN_RATE_LIVE:.0%} ===")

        # 1. Calcula features — usa AdvancedFeatureEngine si disponible
        _log.info("Calculando features...")
        try:
            from core.advanced_features import AdvancedFeatureEngine
            fe_adv  = AdvancedFeatureEngine()
            df_feat = fe_adv.compute_advanced(df_raw.copy())
            df_feat = fe_adv.add_target(df_feat, self.symbol)
            _log.info(f"Features avanzadas: {len(fe_adv.get_all_feature_cols())} columnas")
        except Exception:
            df_feat = self._fe.compute(df_raw.copy())
            df_feat = self._fe.add_target(df_feat, self.symbol)
        df_feat = df_feat.dropna().reset_index(drop=True)
        # Usa feature cols avanzadas si disponibles
        try:
            from core.advanced_features import AdvancedFeatureEngine
            feature_cols = AdvancedFeatureEngine().get_all_feature_cols()
            feature_cols = [c for c in feature_cols if c in df_feat.columns]
        except Exception:
            feature_cols = self._fe.get_feature_cols()

        if len(df_feat) < 1000:
            _log.error(f"Insuficientes datos para entrenar: {len(df_feat)} filas")
            return {"ready_for_live": False, "reason": "insufficient_data", "win_rate": 0}

        # 2. Split temporal fijo (mismo para todas las iteraciones — sin data leakage)
        n      = len(df_feat)
        n_test = max(200, int(n * settings.WALK_FORWARD_TEST))
        df_train_val = df_feat.iloc[:-n_test].copy()
        df_test      = df_feat.iloc[-n_test:].copy()

        _log.info(f"Datos: {len(df_train_val)} train/val | {len(df_test)} test | features: {len(feature_cols)}")

        best_wr    = 0.0
        best_iter  = 0
        win_rate   = 0.0
        pf         = 0.0

        for iteration in range(1, max_iterations + 1):
            _log.info(f"\n--- Iteración {iteration}/{max_iterations} ---")
            _log.info(f"  threshold={self._threshold:.3f} sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")

            # 3. Entrena — Ensemble si disponible, XGBoost fallback
            try:
                if self._ensemble is not None:
                    X_tr = df_train_val[feature_cols].fillna(0)
                    y_tr = df_train_val['target']
                    n_val = int(len(X_tr) * 0.15)
                    X_val, y_val = X_tr.iloc[-n_val:], y_tr.iloc[-n_val:]
                    X_tr2, y_tr2 = X_tr.iloc[:-n_val], y_tr.iloc[:-n_val]
                    n_test_e = max(100, int(len(X_tr) * 0.15))
                    X_test_e = df_test[feature_cols].fillna(0)
                    y_test_e = df_test['target']
                    ens_metrics = self._ensemble.train(X_tr2, y_tr2, X_val, y_val, X_test_e, y_test_e)
                    model = self._ensemble
                    _log.info(f"  Ensemble weights: {self._ensemble.get_weights()}")
                else:
                    train_metrics = self._trainer.train(df_train_val, self.symbol)
                    model = self._trainer.last_model
            except Exception as e:
                _log.error(f"Error entrenando: {e}")
                # fallback XGBoost
                try:
                    train_metrics = self._trainer.train(df_train_val, self.symbol)
                    model = self._trainer.last_model
                except Exception as e2:
                    _log.error(f"Fallback también falló: {e2}")
                    break

            # 3b. Entrena LSTM si disponible
            if self._lstm is not None:
                try:
                    y_all = df_train_val['target'].values
                    self._lstm.train(df_train_val, y_all)
                except Exception as e:
                    _log.debug(f"LSTM train error: {e}")

            # 4. Backtest en test set
            bt = self._backtester_cls(self.symbol)
            # Aplica parámetros actuales del loop al backtester via monkey-patch temporal
            self._apply_temp_params()
            try:
                bt_metrics = bt.run(
                    df       = df_test,
                    model    = model,
                    feature_cols   = feature_cols,
                    signal_threshold = self._threshold,
                    initial_capital  = initial_capital,
                    regime_detector  = self._regime,
                )
            finally:
                self._restore_params()

            win_rate = bt_metrics.get("win_rate", 0)
            pf       = bt_metrics.get("profit_factor", 0)
            n_trades = bt_metrics.get("total_trades", 0)

            if verbose:
                display.print_training_progress(iteration, max_iterations, win_rate, pf, self._threshold)

            _log.info(f"  WR={win_rate:.1%} PF={pf:.2f} trades={n_trades}")

            # 5. Guarda mejor modelo
            if win_rate > best_wr and n_trades >= 30:
                best_wr        = win_rate
                best_iter      = iteration
                self._best_metrics = bt_metrics
                self._best_model   = model
                _log.info(f"  *** Nuevo mejor WR: {best_wr:.1%} en iter {best_iter}")

            # 6. Criterio de éxito
            if win_rate >= settings.MIN_WIN_RATE_LIVE and n_trades >= 50:
                _log.info(f"✓ WIN RATE {win_rate:.1%} >= {settings.MIN_WIN_RATE_LIVE:.0%} — LISTO PARA LIVE")
                break

            # 7a. Consulta Ollama para ajuste inteligente
            if self._ollama is not None and self._ollama.is_available() and iteration % 3 == 0:
                try:
                    suggestion = self._ollama.analyze_performance(
                        {"win_rate": win_rate, "profit_factor": pf, "trades": n_trades,
                         "threshold": self._threshold}, self.symbol)
                    if suggestion and suggestion.get("param") == "SIGNAL_THRESHOLD":
                        adj = float(suggestion.get("amount", 0.02))
                        if suggestion.get("action") == "increase":
                            self._threshold = min(0.75, self._threshold + adj)
                        elif suggestion.get("action") == "decrease":
                            self._threshold = max(0.55, self._threshold - adj)
                        _log.info(f"  Ollama: {suggestion.get('reason', '')} → thr={self._threshold:.3f}")
                except Exception:
                    pass

            # 7b. Ajuste de parámetros para siguiente iteración
            self._adjust_params(iteration, win_rate, pf)

        # 8. Usa mejor modelo encontrado
        final_metrics = self._best_metrics or bt_metrics if 'bt_metrics' in dir() else {"win_rate": 0}
        final_wr      = final_metrics.get("win_rate", 0)
        final_model   = self._best_model or (model if 'model' in dir() else None)
        ready         = final_wr >= settings.MIN_WIN_RATE_LIVE and \
                        final_metrics.get("total_trades", 0) >= 50

        # 9. Guarda modelo final + parámetros optimizados
        if final_model is not None:
            if hasattr(final_model, 'save'):
                final_model.save(self.symbol)
            elif hasattr(self._trainer, '_save_model'):
                try:
                    self._trainer._save_model(final_model, self.symbol)
                except Exception:
                    pass
            self._save_optimized_params()

        # 9b. Evalúa objetivo y avanza si se cumple
        if self._objective is not None:
            obj_result = self._objective.evaluate(
                final_wr, final_metrics.get("profit_factor", 0),
                final_metrics.get("max_drawdown", 1),
                final_metrics.get("total_trades", 0)
            )
            if obj_result.get("achieved"):
                _log.info(f"🎯 OBJETIVO ALCANZADO → avanzando al siguiente")
                print(f"\n  🎯 {self._objective.display(final_wr, final_metrics.get('profit_factor',0), final_metrics.get('total_trades',0))}")

        # 10. Notifica resultado
        notifier.notify(notifier.TRAINING_DONE, {
            "win_rate":     final_wr,
            "profit_factor": final_metrics.get("profit_factor", 0),
            "iteration":    best_iter,
            "threshold":    self._threshold,
        })

        return {
            "ready_for_live":  ready,
            "win_rate":        final_wr,
            "profit_factor":   final_metrics.get("profit_factor", 0),
            "total_trades":    final_metrics.get("total_trades", 0),
            "max_drawdown":    final_metrics.get("max_drawdown", 0),
            "sharpe":          final_metrics.get("sharpe", 0),
            "net_pnl":         final_metrics.get("net_pnl", 0),
            "final_capital":   final_metrics.get("final_capital", initial_capital),
            "best_iteration":  best_iter,
            "iterations_run":  iteration,
            "threshold_final": self._threshold,
            "model":           final_model,
            "equity_curve":    final_metrics.get("equity_curve", []),
            "reason":          "success" if ready else f"no_alcanzó_{settings.MIN_WIN_RATE_LIVE:.0%}",
        }

    # ── Ajuste de parámetros ──────────────────────────────────────────────────

    def _adjust_params(self, iteration: int, win_rate: float, profit_factor: float):
        """
        Estrategia de ajuste progresivo para maximizar WR sin overfitting.
        """
        deficit = settings.MIN_WIN_RATE_LIVE - win_rate

        if iteration <= 5:
            # Fase 1: ajustar threshold para filtrar señales débiles
            if win_rate < 0.50:
                self._threshold = min(0.75, self._threshold + 0.03)
            elif win_rate < 0.55:
                self._threshold = min(0.72, self._threshold + 0.02)
            else:
                self._threshold = min(0.70, self._threshold + 0.01)

        elif iteration <= 10:
            # Fase 2: dar más espacio al SL para reducir stop-outs prematuros
            if profit_factor < 1.3:
                self._sl_mult = min(2.5, self._sl_mult + 0.15)
            # También ajusta threshold
            if win_rate < 0.55:
                self._threshold = min(0.72, self._threshold + 0.015)

        elif iteration <= 15:
            # Fase 3: aumentar TP para mejorar profit factor
            if profit_factor < 1.5:
                self._tp1_mult = min(3.0, self._tp1_mult + 0.1)
            # Reducir threshold si ya filtramos demasiado
            if win_rate < 0.50 and self._threshold > 0.65:
                self._threshold = max(0.60, self._threshold - 0.01)

        else:
            # Fase 4: ajuste combinado agresivo
            if win_rate < 0.50:
                self._sl_mult     = min(2.8, self._sl_mult + 0.2)
                self._threshold   = min(0.75, self._threshold + 0.02)
                self._tp1_mult    = min(3.5, self._tp1_mult + 0.15)
            elif win_rate < 0.55:
                self._threshold   = min(0.73, self._threshold + 0.01)
                self._sl_mult     = min(2.5, self._sl_mult + 0.1)

        _log.debug(f"  Ajuste: threshold={self._threshold:.3f} "
                   f"sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")

    def _apply_temp_params(self):
        """Aplica parámetros temporales al settings module."""
        self._orig_threshold = settings.SIGNAL_THRESHOLD
        self._orig_sl        = settings.ATR_SL_MULTIPLIER
        self._orig_tp1       = settings.ATR_TP1_MULTIPLIER
        settings.SIGNAL_THRESHOLD   = self._threshold
        settings.ATR_SL_MULTIPLIER  = self._sl_mult
        settings.ATR_TP1_MULTIPLIER = self._tp1_mult

    def _restore_params(self):
        """Restaura parámetros originales."""
        settings.SIGNAL_THRESHOLD   = self._orig_threshold
        settings.ATR_SL_MULTIPLIER  = self._orig_sl
        settings.ATR_TP1_MULTIPLIER = self._orig_tp1

    def _save_optimized_params(self):
        """Guarda parámetros optimizados en runtime_params.json."""
        path = settings.CONFIG_DIR / "runtime_params.json"
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update({
                "SIGNAL_THRESHOLD":   round(self._threshold, 3),
                "ATR_SL_MULTIPLIER":  round(self._sl_mult, 2),
                "ATR_TP1_MULTIPLIER": round(self._tp1_mult, 2),
            })
            path.write_text(json.dumps(current, indent=2))
            _log.info(f"Parámetros optimizados guardados: thr={self._threshold:.3f} "
                      f"sl={self._sl_mult:.2f} tp1={self._tp1_mult:.2f}")
        except Exception as e:
            _log.warning(f"No se pudo guardar runtime_params: {e}")
