"""
Orquestador principal. Corre en Python nativo (Linux o Windows); habla con MT5
via `mt5_connector` (API directa por default, o el bridge de archivos hacia
`expert_advisor.mq5` si USE_EA_BRIDGE=True en config.py).

Ciclo por iteracion:
  1. Detecta capital real de la cuenta (capital_detector).
  2. Descarga velas M1 recientes y calcula ATR.
  3. Sincroniza posiciones abiertas reales (por si el proceso se reinicio o
     hay posiciones que no origino este proceso).
  4. Gestiona ganancias de posiciones abiertas (profit_manager: TP escalonado,
     trailing, cierre en maxima ganancia) y niveles de grid pendientes.
  5. Si hay cupo (riesgo, maximo de posiciones, trades/dia) y hay confluencia
     de estrategias, calcula SL/TP con ATR, dimensiona el volumen segun
     capital real, valida margen, y abre la posicion.

DRY_RUN=True (default en config.py) calcula y loguea todo SIN enviar ordenes
reales — usalo para validar el comportamiento antes de operar con dinero real.
"""
import csv
import logging
import time

import ai_optimizer
import capital_detector
import config
import grid_trader
import profit_manager as profit_manager_module
import risk_management
import stop_loss_calculator
from ai_optimizer import TradeStatsSummary
from indicators import atr as atr_indicator
from mt5_connector.connector import MT5Connector, MT5ConnectionError
from strategies import confluence_engine
from strategies.base import Direction

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _log_trade_row(row: dict) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = config.TRADES_LOG_PATH.exists()
    with open(config.TRADES_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class TradingBot:
    def __init__(self):
        self.connector = MT5Connector()
        self.risk_manager = risk_management.RiskManager()
        self.profit_manager = profit_manager_module.ProfitManager()
        self.grid_sessions: dict = {}      # ticket base -> GridSession
        self.known_tickets: set = set()
        self.closed_trades_count = 0
        self.recent_trade_pnls: list = []

    def start(self) -> None:
        self.connector.connect()
        logger.info("Bot iniciado | symbol=%s timeframe=%s DRY_RUN=%s USE_EA_BRIDGE=%s",
                    config.SYMBOL, config.TIMEFRAME, config.DRY_RUN, config.USE_EA_BRIDGE)
        while True:
            try:
                self._loop_once()
            except MT5ConnectionError as e:
                logger.error("Error de conexion MT5: %s", e)
            except Exception:
                logger.exception("Error inesperado en el loop principal (se continua)")
            time.sleep(config.LOOP_INTERVAL_SEC)

    # ------------------------------------------------------------------
    def _loop_once(self) -> None:
        capital_state = capital_detector.detect_capital(self.connector)
        if not capital_state.is_healthy:
            logger.warning("Capital no saludable (equity/margen libre <= 0); se omite ciclo")
            return

        df = self.connector.get_rates()
        current_atr = atr_indicator(df, config.ATR_PERIOD).iloc[-1]
        symbol_info = self.connector.get_symbol_info()
        positions = self.connector.get_open_positions()
        current_tickets = {p.ticket for p in positions}

        self._reconcile_closed_positions(current_tickets)
        self._manage_open_positions(positions, current_atr)

        can_trade, reason = self.risk_manager.can_open_new_trade(capital_state.equity, len(positions))
        if can_trade:
            spread_points = self.connector.get_spread_points()
            result = confluence_engine.evaluate_confluence(df)
            if result.is_actionable:
                self._try_open_trade(result, df, symbol_info, capital_state, spread_points, current_atr)
        else:
            logger.debug("Sin nuevo trade este ciclo: %s", reason)

        self.known_tickets = current_tickets

    # ------------------------------------------------------------------
    def _reconcile_closed_positions(self, current_tickets: set) -> None:
        closed = self.known_tickets - current_tickets
        for ticket in closed:
            pnl = self.connector.get_position_realized_pnl(ticket)
            self.recent_trade_pnls.append(pnl)
            self.closed_trades_count += 1
            self.profit_manager.forget_position(ticket)
            self.grid_sessions.pop(ticket, None)
            _log_trade_row({
                "ticket": ticket, "pnl": round(pnl, 2),
                "closed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            logger.info("Posicion cerrada ticket=%d pnl=%.2f", ticket, pnl)

        if (config.AI_OPTIMIZER_ENABLED and self.closed_trades_count > 0
                and self.closed_trades_count % config.AI_OPTIMIZER_REVIEW_EVERY_N_TRADES == 0
                and closed):
            self._run_ai_review()

    def _run_ai_review(self) -> None:
        pnls = self.recent_trade_pnls[-config.AI_OPTIMIZER_REVIEW_EVERY_N_TRADES:]
        if not pnls:
            return
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        stats = TradeStatsSummary(
            n_trades=len(pnls),
            win_rate=len(wins) / len(pnls) if pnls else 0.0,
            profit_factor=(sum(wins) / -sum(losses)) if losses else (float("inf") if wins else 0.0),
            avg_rr=0.0,  # requeriria el RR planeado por trade; se deja en 0 si no se trackea aparte
            max_drawdown_pct=0.0,
        )
        suggestions = ai_optimizer.request_suggestions(stats)
        if suggestions:
            logger.info("ai_optimizer sugiere (NO aplicado automaticamente, revisar logs/ai_suggested_params.json): %s",
                        suggestions)

    # ------------------------------------------------------------------
    def _manage_open_positions(self, positions, current_atr: float) -> None:
        for pos in positions:
            if pos.ticket not in self.known_tickets:
                self._sync_external_position(pos, current_atr)

            current_price = pos.price_current
            actions = self.profit_manager.evaluate(pos.ticket, current_price, pos.sl, current_atr)
            for action in actions:
                self._apply_profit_action(pos, action)

            grid_session = self.grid_sessions.get(pos.ticket)
            if grid_session is not None:
                self._check_grid_trigger(pos, grid_session, current_price)

    def _sync_external_position(self, pos, current_atr: float) -> None:
        """Registra en profit_manager una posicion que el proceso no origino
        (reinicio del bot, o abierta por otro medio) para que igual reciba
        gestion de ganancias/trailing. Los TP se estiman con ATR actual ya
        que no conocemos el ATR real al momento de apertura."""
        direction = 1 if pos.type == 0 else -1  # MT5: POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1
        tp1 = pos.price_open + direction * current_atr * config.ATR_TP1_MULTIPLIER
        tp2 = pos.price_open + direction * current_atr * config.ATR_TP2_MULTIPLIER
        self.profit_manager.register_position(pos.ticket, pos.price_open, direction, pos.volume, tp1, tp2)
        logger.info("Posicion externa sincronizada: ticket=%d direccion=%d volumen=%.2f",
                    pos.ticket, direction, pos.volume)

    def _apply_profit_action(self, pos, action) -> None:
        if action.type == profit_manager_module.ActionType.PARTIAL_CLOSE:
            self.connector.close_position(pos, volume=action.volume, comment=action.reason[:31])
            logger.info("Cierre parcial ticket=%d vol=%.2f motivo=%s", pos.ticket, action.volume, action.reason)
        elif action.type == profit_manager_module.ActionType.MODIFY_SL:
            self.connector.modify_sltp(pos, sl=action.sl_price)
        elif action.type == profit_manager_module.ActionType.FULL_CLOSE:
            self.connector.close_position(pos, comment=action.reason[:31])
            logger.info("Cierre total ticket=%d motivo=%s", pos.ticket, action.reason)
            self.grid_sessions.pop(pos.ticket, None)

    def _check_grid_trigger(self, pos, grid_session, current_price: float) -> None:
        if grid_session.basket_stop_hit(current_price):
            self.connector.close_position(pos, comment="grid basket stop")
            self.profit_manager.forget_position(pos.ticket)
            self.grid_sessions.pop(pos.ticket, None)
            logger.info("Grid basket stop alcanzado, cerrando ticket=%d", pos.ticket)
            return

        level = grid_session.next_pending_level(current_price)
        if level is None:
            return

        result = self.connector.send_market_order(
            grid_session.direction, level.volume, grid_session.basket_stop_price, None,
            comment=f"grid_lvl_{level.index}",
        )
        opened_ticket = getattr(result, "order", None) if result is not None else None
        if opened_ticket or config.DRY_RUN:
            grid_session.mark_opened(level.index, opened_ticket or -1)
            logger.info("Nivel de grid %d abierto (vol=%.2f) para base ticket=%d", level.index, level.volume, pos.ticket)

    # ------------------------------------------------------------------
    def _try_open_trade(self, result, df, symbol_info, capital_state, spread_points, current_atr) -> None:
        tick = self.connector.get_tick()
        entry_price = tick.ask if result.direction == Direction.BUY else tick.bid

        plan = stop_loss_calculator.calculate_stop_loss_plan(
            int(result.direction), entry_price, df, symbol_info, spread_points,
        )
        if not plan.valid:
            logger.debug("Señal descartada por SL invalido: %s (agreeing=%s)",
                         plan.reason, [s.strategy_name for s in result.agreeing])
            return

        volume = risk_management.calculate_position_size(capital_state.risk_capital, plan.sl_distance_price, symbol_info)
        if volume <= 0:
            logger.debug("Volumen calculado 0: capital de riesgo insuficiente para el lote minimo del broker")
            return

        margin_required = risk_management.estimate_margin_required(volume, entry_price, symbol_info, capital_state.leverage)
        margin_ok, margin_reason = self.risk_manager.check_margin(margin_required, capital_state)
        if not margin_ok:
            logger.debug("Trade descartado por margen: %s", margin_reason)
            return

        comment = "+".join(s.strategy_name for s in result.agreeing)[:31]
        order_result = self.connector.send_market_order(int(result.direction), volume, plan.sl_price, plan.tp1_price, comment=comment)
        ticket = getattr(order_result, "order", None) if order_result is not None else None

        if ticket is None and not config.DRY_RUN:
            logger.warning("Orden no confirmada por el broker, no se registra seguimiento")
            return

        effective_ticket = ticket if ticket is not None else -1  # DRY_RUN: ticket simulado
        self.risk_manager.register_trade_opened()
        self.profit_manager.register_position(effective_ticket, plan.entry_price, int(result.direction),
                                               volume, plan.tp1_price, plan.tp2_price)

        if config.GRID_ENABLED:
            grid_session = grid_trader.build_grid_session(
                int(result.direction), plan.entry_price, plan.atr_value, volume,
                plan.sl_distance_price, symbol_info, capital_state.equity,
            )
            if grid_session.levels:
                self.grid_sessions[effective_ticket] = grid_session

        logger.info("Trade abierto dir=%s vol=%.2f entry=%.5f sl=%.5f tp1=%.5f tp2=%.5f rr=%.2f confluencia=%s",
                    result.direction.name, volume, plan.entry_price, plan.sl_price, plan.tp1_price,
                    plan.tp2_price, plan.risk_reward_ratio, comment)


if __name__ == "__main__":
    TradingBot().start()
