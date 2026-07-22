import config
from grid_trader import build_grid_session


def test_grid_levels_are_below_entry_for_buy(symbol_info):
    session = build_grid_session(
        direction=1, entry_price=1900.0, atr_value=2.0, base_volume=0.01,
        sl_distance_price=3.0, symbol_info=symbol_info, equity=1000.0,
    )
    assert len(session.levels) <= config.GRID_LEVELS
    for level in session.levels:
        assert level.trigger_price < 1900.0
        assert level.volume > 0
    assert session.basket_stop_price < session.levels[-1].trigger_price if session.levels else True


def test_grid_levels_are_above_entry_for_sell(symbol_info):
    session = build_grid_session(
        direction=-1, entry_price=1900.0, atr_value=2.0, base_volume=0.01,
        sl_distance_price=3.0, symbol_info=symbol_info, equity=1000.0,
    )
    for level in session.levels:
        assert level.trigger_price > 1900.0
    assert session.basket_stop_price > session.levels[-1].trigger_price if session.levels else True


def test_grid_respects_max_total_risk_cap(symbol_info):
    # equity muy chico fuerza a que el grid se corte antes de GRID_LEVELS
    session = build_grid_session(
        direction=1, entry_price=1900.0, atr_value=5.0, base_volume=1.0,
        sl_distance_price=10.0, symbol_info=symbol_info, equity=50.0,
    )
    assert len(session.levels) < config.GRID_LEVELS


def test_next_pending_level_triggers_on_price_drop(symbol_info):
    session = build_grid_session(
        direction=1, entry_price=1900.0, atr_value=2.0, base_volume=0.01,
        sl_distance_price=3.0, symbol_info=symbol_info, equity=1000.0,
    )
    assert session.levels, "se esperaban niveles de grid con este capital"
    first_trigger = session.levels[0].trigger_price
    assert session.next_pending_level(first_trigger - 0.01) is not None
    assert session.next_pending_level(1900.0) is None
