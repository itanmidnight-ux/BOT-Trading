"""
Loop principal de trading en vivo.
Event-driven via MT5Stream: actúa exactamente al cierre de cada vela M1.
"""
import time
from datetime import datetime
from typing import Optional

from utils.logger import get_logger
from utils import display, notifier
from config import settings, constants

_log = get_logger("live_trader")


class LiveTrader:

    def __init__(self, symbol: str, mt5_connector, mt5_stream,
                 feature_engine, model_updater, regime_detector,
                 mtf_analyzer, signal_generator, kelly_engine,
                 risk_manager, trade_manager, exit_manager,
                 state_manager, data_updater,
                 auto_improver):
        self.symbol         = symbol
        self._conn          = mt5_connector
        self._stream        = mt5_stream
        self._fe            = feature_engine
        self._model_upd     = model_updater
        self._regime        = regime_detector
        self._mtf           = mtf_analyzer
        self._signal        = signal_generator
        self._kelly         = kelly_engine
        self._risk          = risk_manager
        self._trades        = trade_manager
        self._exit          = exit_manager
        self._state         = state_manager
        self._data_upd      = data_updater
        self._improver      = auto_improver

        self._running       = False
        self._trades_today  = 0
        self._wins_today    = 0
        self._pnl_today     = 0.0
        self._bar_count     = 0

    def start(self):
        """Inicia el loop de trading en vivo."""
        _log.info(f"=== LIVE TRADER INICIADO — {self.symbol} ===")
        self._running = True

        # Verificar y sincronizar estado
        self._state.verify_with_mt5(self._trades)

        # Suscribir al stream de velas M1
        self._stream.start(self.symbol, settings.TIMEFRAME_MAIN, self._on_bar_close)

        _log.info("Stream M1 activo. Esperando cierre de velas...")
        print(f"\n[BOT] Esperando velas {self.symbol}... (Ctrl+C para detener)\n")

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Para el bot limpiamente."""
        _log.info("Deteniendo bot...")
        self._running = False
        self._stream.stop()

        # Cierra posiciones abiertas
        open_pos = self._trades.get_open_positions(self.symbol)
        if open_pos:
            _log.info(f"Cerrando {len(open_pos)} posición(es) abiertas...")
            self._trades.close_all_positions(self.symbol)

        print("\n[BOT] Bot detenido limpiamente.")
        _log.info("Bot detenido.")

    def _on_bar_close(self, bar: dict):
        """
        Callback disparado al cierre de cada vela M1.
        Secuencia: datos → features → régimen → exit check → nueva señal → orden
        """
        self._bar_count += 1

        try:
            # 1. Actualizar datos
            self._data_upd.update_one(self.symbol, settings.TIMEFRAME_MAIN)
            df = self._stream.get_latest_bars(self.symbol, settings.TIMEFRAME_MAIN, 300)
            if df is None or len(df) < 60:
                return

            # 2. Calcular features
            df_feat      = self._fe.compute(df.copy())
            feature_cols = self._fe.get_feature_cols()
            if df_feat is None or len(df_feat) < 50:
                return

            # 3. ATR actual
            atr      = float(df_feat["atr"].iloc[-1]) if "atr" in df_feat.columns else 0
            atr_mean = float(df_feat["atr_mean50"].iloc[-1]) if "atr_mean50" in df_feat.columns else atr

            # 4. Actualizar multi-TF
            for tf in settings.TIMEFRAMES_CONFIRM:
                df_tf = self._stream.get_latest_bars(self.symbol, tf, 100)
                if df_tf is not None and len(df_tf) >= 30:
                    df_tf_feat = self._fe.compute(df_tf.copy())
                    self._mtf.update(self.symbol, tf, df_tf_feat)

            # 5. Cuenta info
            account = self._conn.get_account_info()
            equity  = account.get("equity", self._state.capital)
            free_margin = account.get("free_margin", equity)
            margin  = account.get("margin", 0)

            # 6. Gestión de posición abierta
            if self._state.has_open_position(self.symbol):
                self._manage_open_position(bar, atr, equity)

            # 7. Nueva señal (si no hay posición abierta)
            if not self._state.has_open_position(self.symbol):
                safe, reason = self._risk.check_all(
                    self.symbol, equity, atr, atr_mean, free_margin, margin
                )
                if not safe:
                    if self._bar_count % 10 == 0:
                        _log.debug(f"Risk check failed: {reason}")
                else:
                    signal = self._signal.generate(self.symbol, df_feat, feature_cols)
                    if signal and signal.direction != constants.SIGNAL_HOLD:
                        self._execute_signal(signal, equity, free_margin)

            # 9. Dashboard
            pos_info = "Sin posicion"
            pos      = self._state.get_position(self.symbol)
            if pos:
                entry   = pos.get("entry", 0)
                tick    = self._stream.get_latest_tick(self.symbol)
                cur_p   = tick.get("bid", entry) if tick else entry
                diff    = cur_p - entry if pos["direction"] == constants.SIGNAL_BUY else entry - cur_p
                pnl_est = diff / 0.00010 * pos["lots"] * 10 if "XAU" not in self.symbol else diff * pos["lots"] * 100
                pos_info = f"{pos['direction']} {pos['lots']}lots ~${pnl_est:+.2f}"

            display.update_dashboard(
                self.symbol, self._state.capital, equity,
                self._pnl_today, self._trades_today, self._wins_today,
                self._state.phase, pos_info
            )

            # 10. Auto-mejora cada 100 velas
            if self._bar_count % 100 == 0:
                changes = self._improver.run(self.symbol)
                if changes:
                    _log.info(f"Auto-mejora aplicada: {changes}")

            # 11. Re-entrenamiento si necesario
            if self._data_upd.should_retrain(self.symbol):
                _log.info("Trigger re-entrenamiento...")
                self._model_upd.maybe_retrain(self.symbol, df_feat, self._data_upd)

        except Exception as e:
            _log.error(f"Error en on_bar_close: {e}", exc_info=True)

    def _manage_open_position(self, bar: dict, atr: float, equity: float):
        """
        Evalúa la posición abierta. Ciclo de 1 vela: evaluate() siempre
        devuelve CLOSE_FULL (por SL, TP, o forzado al cierre de la vela).
        """
        pos_data = self._state.get_position(self.symbol)
        if pos_data is None:
            return

        from core.exit_manager import OpenPosition
        position = OpenPosition(
            ticket     = pos_data["ticket"],
            symbol     = self.symbol,
            direction  = pos_data["direction"],
            lots       = pos_data["lots"],
            entry      = pos_data["entry"],
            sl         = pos_data["sl"],
            tp         = pos_data["tp"],
            open_time  = datetime.fromisoformat(pos_data["open_time"]) if isinstance(pos_data["open_time"], str) else pos_data["open_time"],
            bars_open  = pos_data.get("bars_open", 0),
        )

        action = self._exit.evaluate(position, bar, atr)

        success = self._trades.close_position(
            position.ticket, self.symbol, position.direction
        )
        if success:
            tick    = self._stream.get_latest_tick(self.symbol)
            cur_p   = tick.get("bid" if position.direction == constants.SIGNAL_BUY else "ask", position.entry) if tick else position.entry
            diff    = cur_p - position.entry if position.direction == constants.SIGNAL_BUY else position.entry - cur_p
            pips    = diff / 0.1
            pnl     = pips * position.lots * 100

            win = pnl > 0
            self._pnl_today += pnl
            self._trades_today += 1
            if win:
                self._wins_today += 1

            self._state.record_trade_result(win)
            self._state.capital = max(0, self._state.capital + pnl)
            self._state.clear_position(self.symbol)
            self._kelly.update(self.symbol, win, abs(pips))

            notifier.notify(notifier.TRADE_CLOSE, {
                "symbol":    self.symbol,
                "direction": position.direction,
                "pips":      round(pips, 1),
                "pnl":       round(pnl, 4),
                "reason":    action.reason,
            })

    def _execute_signal(self, signal, equity: float, free_margin: float):
        """Ejecuta una señal: calcula lots, verifica margen, coloca orden."""
        # Precio actual como entrada estimada
        tick   = self._stream.get_latest_tick(self.symbol)
        if tick is None:
            return
        entry  = tick.get("ask" if signal.direction == constants.SIGNAL_BUY else "bid", 0)
        if entry <= 0:
            return

        levels = self._exit.calc_levels(self.symbol, signal.direction, entry, signal.atr)

        # Kelly position sizing
        self._kelly.recalculate(self.symbol)
        fraction = self._kelly.get_current_fraction(self.symbol)
        symbol_info = self._conn.get_symbol_info(self.symbol)
        lots = self._kelly.fraction_to_lots(fraction, equity, self.symbol, entry)

        if not self._kelly.verify_margin(lots, self.symbol, free_margin, symbol_info):
            _log.warning(f"Margen insuficiente para {lots} lots — reduciendo a mínimo")
            lots = settings.KELLY_MIN_LOTS

        result = self._trades.open_market_order(
            symbol    = self.symbol,
            direction = signal.direction,
            lots      = lots,
            sl_price  = levels["sl"],
            tp_price  = levels["tp"],
        )

        if result.success:
            self._state.save_position(
                symbol    = self.symbol,
                ticket    = result.ticket,
                direction = signal.direction,
                lots      = lots,
                entry     = result.entry,
                sl        = levels["sl"],
                tp        = levels["tp"],
            )
            notifier.notify(notifier.TRADE_OPEN, {
                "symbol":    self.symbol,
                "direction": signal.direction,
                "lots":      lots,
                "entry":     result.entry,
                "sl":        levels["sl"],
                "tp":        levels["tp"],
            })
        else:
            _log.warning(f"Orden fallida: {result.message}")
