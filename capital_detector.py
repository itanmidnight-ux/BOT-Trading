"""
Deteccion automatica de capital disponible.

No asume ningun monto fijo: en cada ciclo consulta balance/equity/margen libre
reales de la cuenta MT5 y deriva de ahi cuanto capital de riesgo hay disponible
en ese momento. Esto es lo que le permite al bot funcionar igual con $50 que
con $50,000 sin tocar codigo — solo cambia el lotaje resultante.
"""
import logging
from dataclasses import dataclass

import config
from mt5_connector.connector import MT5Connector

logger = logging.getLogger(__name__)


@dataclass
class CapitalState:
    balance: float
    equity: float
    margin_free: float
    risk_capital: float        # equity * RISK_PER_TRADE_PCT, ya clampeado al techo duro
    max_allocatable_margin: float  # equity * MAX_CAPITAL_ALLOCATION_PCT, techo de margen usable
    currency: str
    leverage: int

    @property
    def is_healthy(self) -> bool:
        """False si el margen libre no alcanza para operar con seguridad."""
        return self.margin_free > 0 and self.equity > 0


def detect_capital(connector: MT5Connector) -> CapitalState:
    snapshot = connector.get_account_snapshot()

    risk_pct = min(config.RISK_PER_TRADE_PCT, config.MAX_RISK_PER_TRADE_PCT)
    risk_capital = snapshot.equity * (risk_pct / 100.0)
    max_allocatable_margin = snapshot.equity * (config.MAX_CAPITAL_ALLOCATION_PCT / 100.0)

    state = CapitalState(
        balance=snapshot.balance,
        equity=snapshot.equity,
        margin_free=snapshot.margin_free,
        risk_capital=risk_capital,
        max_allocatable_margin=max_allocatable_margin,
        currency=snapshot.currency,
        leverage=snapshot.leverage,
    )

    if not state.is_healthy:
        logger.warning("Estado de capital no saludable: equity=%.2f margin_free=%.2f",
                        state.equity, state.margin_free)

    return state
