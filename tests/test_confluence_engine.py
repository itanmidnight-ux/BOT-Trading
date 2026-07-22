import config
from strategies.confluence_engine import evaluate_confluence, run_all_strategies


def test_insufficient_data_is_flat(make_df):
    df = make_df(n=10)
    result = evaluate_confluence(df)
    assert not result.is_actionable


def test_run_all_strategies_returns_one_signal_per_enabled_strategy(trending_df):
    signals = run_all_strategies(trending_df)
    enabled_count = sum(1 for v in config.STRATEGIES_ENABLED.values() if v)
    assert len(signals) == enabled_count


def test_confluence_result_serializes_to_dict(trending_df):
    result = evaluate_confluence(trending_df)
    payload = result.as_dict()
    assert "direction" in payload
    assert "all_signals" in payload
    assert len(payload["all_signals"]) == sum(1 for v in config.STRATEGIES_ENABLED.values() if v)
