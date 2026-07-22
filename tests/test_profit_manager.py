from profit_manager import ActionType, ProfitManager


def test_tp1_triggers_partial_close():
    pm = ProfitManager()
    pm.register_position(1, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1905.0, tp2_price=1950.0)

    actions = pm.evaluate(1, current_price=1905.5, current_sl=1895.0, atr_value=2.0)

    assert any(a.type == ActionType.PARTIAL_CLOSE for a in actions)


def test_no_actions_while_in_loss():
    pm = ProfitManager()
    pm.register_position(1, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1905.0, tp2_price=1950.0)

    actions = pm.evaluate(1, current_price=1898.0, current_sl=1895.0, atr_value=2.0)

    assert actions == []


def test_trailing_arms_and_closes_on_giveback():
    pm = ProfitManager()
    pm.register_position(2, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1950.0, tp2_price=1980.0)

    # Profit grande -> arma trailing y mueve el SL a favor.
    first = pm.evaluate(2, current_price=1910.0, current_sl=1895.0, atr_value=2.0)
    assert any(a.type == ActionType.MODIFY_SL for a in first)

    # El precio retrocede fuerte desde el pico -> debe cerrar en la maxima ganancia detectada.
    second = pm.evaluate(2, current_price=1900.5, current_sl=1895.0, atr_value=2.0)
    assert any(a.type == ActionType.FULL_CLOSE for a in second)


def test_forget_position_clears_tracker():
    pm = ProfitManager()
    pm.register_position(3, entry_price=1900.0, direction=1, volume=0.10, tp1_price=1905.0, tp2_price=1950.0)
    pm.forget_position(3)

    actions = pm.evaluate(3, current_price=1910.0, current_sl=1895.0, atr_value=2.0)

    assert actions == []
