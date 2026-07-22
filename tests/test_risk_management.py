import config
from risk_management import RiskManager, calculate_position_size, estimate_margin_required


def test_position_size_scales_with_risk_capital(symbol_info):
    small = calculate_position_size(10.0, sl_distance_price=5.0, symbol_info=symbol_info)
    large = calculate_position_size(100.0, sl_distance_price=5.0, symbol_info=symbol_info)
    assert large > small


def test_position_size_zero_when_below_broker_min_lot(symbol_info):
    volume = calculate_position_size(0.01, sl_distance_price=50.0, symbol_info=symbol_info)
    assert volume == 0.0


def test_position_size_zero_for_invalid_inputs(symbol_info):
    assert calculate_position_size(0.0, sl_distance_price=5.0, symbol_info=symbol_info) == 0.0
    assert calculate_position_size(10.0, sl_distance_price=0.0, symbol_info=symbol_info) == 0.0


def test_daily_loss_gate_blocks_trading():
    rm = RiskManager()
    rm.update(1000.0)
    equity_after_loss = 1000.0 * (1 - (config.MAX_DAILY_LOSS_PCT + 1) / 100.0)

    allowed, reason = rm.can_open_new_trade(equity_after_loss, open_positions_count=0)

    assert allowed is False
    assert "perdida diaria" in reason


def test_drawdown_kill_switch_blocks_trading():
    rm = RiskManager()
    rm.update(1000.0)
    rm.update(1200.0)  # nuevo peak
    equity_after_drawdown = 1200.0 * (1 - (config.MAX_DRAWDOWN_PCT + 1) / 100.0)

    allowed, reason = rm.can_open_new_trade(equity_after_drawdown, open_positions_count=0)

    assert allowed is False
    assert "drawdown" in reason


def test_max_open_positions_gate():
    rm = RiskManager()
    rm.update(1000.0)

    allowed, reason = rm.can_open_new_trade(1000.0, open_positions_count=config.MAX_OPEN_POSITIONS)

    assert allowed is False
    assert "posiciones abiertas" in reason


def test_margin_estimation_is_positive(symbol_info):
    margin = estimate_margin_required(1.0, 1900.0, symbol_info, leverage=100)
    assert margin > 0
