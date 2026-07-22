from indicators import atr, ema, rsi


def test_ema_follows_uptrend(trending_df):
    e = ema(trending_df["close"], 10)
    assert e.iloc[-1] > e.iloc[10]


def test_rsi_is_bounded(trending_df):
    r = rsi(trending_df["close"], 14)
    assert (r.dropna() >= 0).all()
    assert (r.dropna() <= 100).all()


def test_atr_is_non_negative(trending_df):
    a = atr(trending_df, 14)
    assert (a.dropna() >= 0).all()
