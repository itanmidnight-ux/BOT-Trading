"""
Gestion de riesgo: position sizing dinamico + limites duros que ninguna señal
puede saltarse (perdida diaria maxima, drawdown maximo, maximo de posiciones
abiertas). Este modulo es deliberadamente independiente de mt5_connector para
poder testearlo sin conexion real (ver tests/test_risk_management.py).
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import config
from capital_detector import CapitalState

logger = logging.getLogger(__name__)


def calculate_position_size(risk_capital: float, sl_distance_price: float, symbol_info) -> float:
    """Devuelve el volumen (lotes) tal que, si el precio toca el SL, la perdida
    monetaria sea aproximadamente `risk_capital`. Redondea hacia abajo al step
    del simbolo para nunca exceder el riesgo pretendido."""
    if sl_distance_price <= 0 or risk_capital <= 0:
        return 0.0

    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    if tick_size <= 0 or tick_value <= 0:
        return 0.0

    value_per_price_unit_per_lot = tick_value / tick_size
    loss_per_lot = sl_distance_price * value_per_price_unit_per_lot
    if loss_per_lot <= 0:
        return 0.0

    raw_volume = risk_capital / loss_per_lot

    step = symbol_info.volume_step or config.MIN_LOT
    volume = math.floor(raw_volume / step) * step
    volume = max(volume, 0.0)
    volume = min(volume, symbol_info.volume_max)
    if volume < symbol_info.volume_min:
        return 0.0  # el capital/riesgo actual no alcanza ni para el lote minimo del broker

    return round(volume, 2)


def estimate_margin_required(volume: float, price: float, symbol_info, leverage: int) -> float:
    contract_size = symbol_info.trade_contract_size
    if leverage <= 0:
        return float("inf")
    return (volume * contract_size * price) / leverage


@dataclass
class DailyRiskState:
    day: Optional[str] = None
    starting_equity: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0

    def reset_if_new_day(self, equity: float):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.day != today:
            self.day = today
            self.starting_equity = equity
            self.peak_equity = equity
            self.trades_today = 0
            logger.info("Nuevo dia de trading (%s), equity inicial=%.2f", today, equity)


class RiskManager:
    def __init__(self):
        self.state = DailyRiskState()

    def update(self, equity: float):
        self.state.reset_if_new_day(equity)
        self.state.peak_equity = max(self.state.peak_equity, equity)

    def register_trade_opened(self):
        self.state.trades_today += 1

    # ------------------------------------------------------------------
    # Gates de seguridad
    # ------------------------------------------------------------------
    def daily_loss_exceeded(self, equity: float) -> bool:
        if self.state.starting_equity <= 0:
            return False
        loss_pct = (self.state.starting_equity - equity) / self.state.starting_equity * 100.0
        return loss_pct >= config.MAX_DAILY_LOSS_PCT

    def drawdown_exceeded(self, equity: float) -> bool:
        if self.state.peak_equity <= 0:
            return False
        dd_pct = (self.state.peak_equity - equity) / self.state.peak_equity * 100.0
        return dd_pct >= config.MAX_DRAWDOWN_PCT

    def max_positions_reached(self, open_positions_count: int) -> bool:
        return open_positions_count >= config.MAX_OPEN_POSITIONS

    def max_trades_today_reached(self) -> bool:
        return self.state.trades_today >= config.MAX_TRADES_PER_DAY

    def can_open_new_trade(self, equity: float, open_positions_count: int) -> tuple[bool, str]:
        self.update(equity)

        if self.daily_loss_exceeded(equity):
            return False, "limite de perdida diaria alcanzado"
        if self.drawdown_exceeded(equity):
            return False, "drawdown maximo alcanzado (kill-switch)"
        if self.max_positions_reached(open_positions_count):
            return False, "maximo de posiciones abiertas alcanzado"
        if self.max_trades_today_reached():
            return False, "maximo de trades diarios alcanzado"
        return True, "ok"

    def check_margin(self, margin_required: float, capital_state: CapitalState) -> tuple[bool, str]:
        if margin_required > capital_state.margin_free:
            return False, "margen libre insuficiente"
        if margin_required > capital_state.max_allocatable_margin:
            return False, "excede asignacion maxima de capital configurada"
        return True, "ok"
