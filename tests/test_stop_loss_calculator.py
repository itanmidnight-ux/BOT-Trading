import config
from stop_loss_calculator import calculate_stop_loss_plan


def test_buy_plan_has_sl_below_entry_and_meets_min_rr(make_df, symbol_info):
    df = make_df(n=100)
    entry = df["close"].iloc[-1]
    plan = calculate_stop_loss_plan(1, entry, df, symbol_info, spread_points=20)

    assert plan.valid
    assert plan.sl_price < entry
    assert plan.tp1_price > entry
    assert plan.tp2_price > plan.tp1_price
    assert plan.risk_reward_ratio >= config.MIN_RR_RATIO


def test_sell_plan_has_sl_above_entry(make_df, symbol_info):
    df = make_df(n=100)
    entry = df["close"].iloc[-1]
    plan = calculate_stop_loss_plan(-1, entry, df, symbol_info, spread_points=20)

    assert plan.valid
    assert plan.sl_price > entry
    assert plan.tp1_price < entry
    assert plan.tp2_price < plan.tp1_price


def test_excessive_spread_invalidates_plan(make_df, symbol_info):
    df = make_df(n=100)
    entry = df["close"].iloc[-1]
    plan = calculate_stop_loss_plan(1, entry, df, symbol_info, spread_points=config.MAX_SPREAD_POINTS + 100)

    assert not plan.valid


def test_insufficient_bars_invalidates_plan(make_df, symbol_info):
    df = make_df(n=5)
    entry = df["close"].iloc[-1]
    plan = calculate_stop_loss_plan(1, entry, df, symbol_info, spread_points=20)

    assert not plan.valid
