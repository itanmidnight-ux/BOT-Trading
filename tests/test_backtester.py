"""No valida rentabilidad (no hay datos historicos reales embebidos, ver
docstring de backtester.py): solo verifica que el motor corre end-to-end
sobre datos sinteticos sin excepciones y produce un reporte consistente."""
from backtester import run_backtest


def test_backtest_runs_end_to_end_without_errors(make_df):
    df = make_df(n=400, seed=7)
    report = run_backtest(df, initial_balance=1000.0)

    assert report.initial_balance == 1000.0
    assert isinstance(report.final_balance, float)
    assert report.total_trades == len(report.trades)
    assert 0.0 <= report.win_rate <= 1.0
    assert len(report.equity_curve) >= 1
