"""
mt5_stream.py
Real-time data layer for MetaTrader 5.

Components:
  - MT5Stream  : daemon thread that detects bar-close events and fires a callback.
  - get_latest_bars()  : snapshot of the last N completed bars as a DataFrame.
  - get_latest_tick()  : latest bid/ask/time for a symbol.

Bar-close detection:
  The thread polls mt5.copy_rates_from_pos() every second.  When the timestamp
  of the newest bar changes compared to the previous poll, the *previous* bar is
  considered closed and the callback receives it as a dict.

Buffer:
  An in-memory deque of up to BUFFER_SIZE (500) bar dicts is maintained per
  (symbol, timeframe) key.  get_latest_bars() prefers this buffer and falls back
  to a live MT5 query when the buffer is too small.

Reconnection:
  If copy_rates_from_pos() returns None the thread waits RECONNECT_WAIT seconds
  and retries indefinitely until stopped.
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from core.mt5_compat import mt5
from config.constants import CANDLE_COLS
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TF_MAP: Dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

BUFFER_SIZE = 500          # max candles kept per (symbol, timeframe)
POLL_INTERVAL = 1.0        # seconds between bar-close checks
RECONNECT_WAIT = 5.0       # seconds to wait after a failed MT5 query

# ---------------------------------------------------------------------------
# Module-level buffer shared across all MT5Stream instances
# ---------------------------------------------------------------------------
# Key: (symbol, timeframe_str)  →  Value: deque of bar dicts (oldest … newest)
_bar_buffers: Dict[tuple, deque] = {}
_buffer_lock = threading.Lock()

log = get_logger("mt5_stream")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_timeframe(timeframe_str: str) -> int:
    """Convert a timeframe string such as 'M1' to the MT5 integer constant."""
    tf = TF_MAP.get(timeframe_str.upper())
    if tf is None:
        raise ValueError(
            "Unknown timeframe '%s'. Supported: %s"
            % (timeframe_str, list(TF_MAP.keys()))
        )
    return tf


def _rates_to_df(rates) -> pd.DataFrame:
    """
    Convert the numpy structured array returned by copy_rates_from_pos()
    to a clean DataFrame with CANDLE_COLS columns and proper datetime index.
    """
    df = pd.DataFrame(rates)
    # Keep only the columns we care about (MT5 also returns 'real_volume')
    cols_present = [c for c in CANDLE_COLS if c in df.columns]
    df = df[cols_present].copy()

    # Convert numpy int64 UTC timestamps to pandas datetime64[ns, UTC]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    return df.reset_index(drop=True)


def _rate_to_dict(rate) -> dict:
    """
    Convert a single row of the MT5 rates structured array to a plain dict.
    The 'time' field is converted to a UTC-aware datetime object.
    """
    d = {col: rate[col] for col in CANDLE_COLS if col in rate.dtype.names}
    if "time" in d:
        ts = int(d["time"])
        d["time"] = datetime.fromtimestamp(ts, tz=timezone.utc)
    return d


def _fetch_rates(symbol: str, tf_int: int, n: int):
    """
    Fetch the last *n* bars from MT5.  Returns the raw rates array or None.
    """
    rates = mt5.copy_rates_from_pos(symbol, tf_int, 0, n)
    if rates is None or len(rates) == 0:
        log.warning(
            "_fetch_rates(%s, tf=%d, n=%d) returned empty. Last error: %s",
            symbol, tf_int, n, mt5.last_error(),
        )
        return None
    return rates


def _update_buffer(key: tuple, bar_dict: dict) -> None:
    """Append *bar_dict* to the buffer for *key*, trimming to BUFFER_SIZE."""
    with _buffer_lock:
        if key not in _bar_buffers:
            _bar_buffers[key] = deque(maxlen=BUFFER_SIZE)
        _bar_buffers[key].append(bar_dict)


def _seed_buffer(symbol: str, timeframe_str: str, tf_int: int) -> None:
    """Pre-fill the buffer with the last BUFFER_SIZE historical bars."""
    key = (symbol, timeframe_str)
    rates = _fetch_rates(symbol, tf_int, BUFFER_SIZE)
    if rates is None:
        log.warning("Could not seed buffer for %s %s.", symbol, timeframe_str)
        return

    with _buffer_lock:
        _bar_buffers[key] = deque(
            (_rate_to_dict(r) for r in rates),
            maxlen=BUFFER_SIZE,
        )
    log.debug("Buffer seeded: %s %s — %d bars.", symbol, timeframe_str, len(_bar_buffers[key]))


# ---------------------------------------------------------------------------
# MT5Stream
# ---------------------------------------------------------------------------

class MT5Stream:
    """
    Daemon thread that streams bar-close events for a single (symbol, timeframe).

    Parameters
    ----------
    symbol : str
        e.g. "EURUSD"
    timeframe_str : str
        One of "M1", "M5", "M15", "H1", "H4".
    on_bar_close_callback : Callable[[dict], None]
        Invoked on the stream thread whenever a bar closes.
        The dict contains: time (datetime UTC), open, high, low, close,
        tick_volume, spread.
    """

    def __init__(
        self,
        symbol: str,
        timeframe_str: str,
        on_bar_close_callback: Callable[[dict], None],
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe_str = timeframe_str.upper()
        self.tf_int = _resolve_timeframe(self.timeframe_str)
        self.callback = on_bar_close_callback
        self._key = (self.symbol, self.timeframe_str)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_bar_time: Optional[int] = None   # raw UTC timestamp (int)

        log.info(
            "MT5Stream created: symbol=%s timeframe=%s",
            self.symbol, self.timeframe_str,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Seed the buffer and start the background thread."""
        _seed_buffer(self.symbol, self.timeframe_str, self.tf_int)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"MT5Stream-{self.symbol}-{self.timeframe_str}",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "MT5Stream started: %s %s (thread=%s)",
            self.symbol, self.timeframe_str, self._thread.name,
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish (up to 10 s)."""
        log.info("Stopping MT5Stream %s %s …", self.symbol, self.timeframe_str)
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                log.warning(
                    "MT5Stream thread %s did not stop within 10 s.",
                    self._thread.name,
                )
        log.info("MT5Stream %s %s stopped.", self.symbol, self.timeframe_str)

    def is_running(self) -> bool:
        """Return True if the stream thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        log.debug(
            "Stream thread running: %s %s", self.symbol, self.timeframe_str
        )
        # Fetch 2 bars: index-0 is the forming bar, index-1 is the last closed bar.
        # We track the timestamp of the forming bar; when it changes, the old
        # forming bar has closed.
        while not self._stop_event.is_set():
            rates = _fetch_rates(self.symbol, self.tf_int, 2)

            if rates is None:
                log.warning(
                    "Stream %s %s: MT5 query failed — reconnecting in %.0f s.",
                    self.symbol, self.timeframe_str, RECONNECT_WAIT,
                )
                self._stop_event.wait(RECONNECT_WAIT)
                continue

            # rates[0] = oldest (last closed bar), rates[1] = forming bar
            # When asking from_pos(0, 2) MT5 returns newest first, so:
            #   rates[0] is the forming/current bar
            #   rates[1] is the previous (closed) bar
            # Verify actual ordering by comparing timestamps.
            if len(rates) < 2:
                self._stop_event.wait(POLL_INTERVAL)
                continue

            # Ensure rates are sorted oldest-first
            if rates[0]["time"] > rates[1]["time"]:
                forming_bar = rates[0]
                closed_bar  = rates[1]
            else:
                forming_bar = rates[1]
                closed_bar  = rates[0]

            current_ts = int(forming_bar["time"])

            if self._last_bar_time is None:
                # First poll — just record the current bar timestamp
                self._last_bar_time = current_ts
                log.debug(
                    "Stream %s %s: initial bar ts=%d",
                    self.symbol, self.timeframe_str, current_ts,
                )
            elif current_ts != self._last_bar_time:
                # The forming bar has rolled over — the previous forming bar closed
                self._last_bar_time = current_ts
                closed_dict = _rate_to_dict(closed_bar)
                _update_buffer(self._key, closed_dict)
                log.debug(
                    "Bar closed: %s %s @ %s  O=%.5f H=%.5f L=%.5f C=%.5f",
                    self.symbol, self.timeframe_str,
                    closed_dict["time"],
                    closed_dict.get("open", 0),
                    closed_dict.get("high", 0),
                    closed_dict.get("low", 0),
                    closed_dict.get("close", 0),
                )
                try:
                    self.callback(closed_dict)
                except Exception as exc:
                    log.error(
                        "on_bar_close_callback raised an exception: %s", exc,
                        exc_info=True,
                    )

            self._stop_event.wait(POLL_INTERVAL)

        log.debug(
            "Stream thread exiting: %s %s", self.symbol, self.timeframe_str
        )


# ---------------------------------------------------------------------------
# Stateless query helpers
# ---------------------------------------------------------------------------

def get_latest_bars(
    symbol: str,
    timeframe_str: str,
    n: int,
) -> pd.DataFrame:
    """
    Return a DataFrame of the last *n* closed bars for *symbol* / *timeframe_str*.

    Column order matches CANDLE_COLS: time, open, high, low, close,
    tick_volume, spread.  The 'time' column contains UTC-aware datetime64 values.

    The in-memory buffer is used when it holds enough bars; otherwise a live
    MT5 query is issued.

    Parameters
    ----------
    symbol : str
    timeframe_str : str  — "M1", "M5", "M15", "H1", "H4"
    n : int             — number of bars requested (≥ 1)

    Returns
    -------
    pd.DataFrame  — rows are sorted oldest → newest; may have fewer than *n*
                    rows if history is unavailable.

    Raises
    ------
    ValueError  — unknown timeframe_str.
    """
    symbol = symbol.upper()
    timeframe_str = timeframe_str.upper()
    key = (symbol, timeframe_str)

    # Try buffer first
    with _buffer_lock:
        buf = _bar_buffers.get(key)
        if buf is not None and len(buf) >= n:
            bars = list(buf)[-n:]
            df = pd.DataFrame(bars, columns=[c for c in CANDLE_COLS if c in bars[0]])
            # Ensure 'time' is datetime64 with UTC tz
            if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(
                    df["time"].apply(lambda t: t.timestamp() if isinstance(t, datetime) else t),
                    unit="s",
                    utc=True,
                )
            return df.reset_index(drop=True)

    # Fall back to live MT5 query
    log.debug(
        "get_latest_bars(%s, %s, %d): buffer miss — querying MT5.",
        symbol, timeframe_str, n,
    )
    tf_int = _resolve_timeframe(timeframe_str)
    # Request n+1 so we can exclude the still-forming bar (index 0)
    rates = _fetch_rates(symbol, tf_int, n + 1)
    if rates is None:
        log.error(
            "get_latest_bars(%s, %s, %d): MT5 query returned None.",
            symbol, timeframe_str, n,
        )
        return pd.DataFrame(columns=CANDLE_COLS)

    # Exclude the forming (newest) bar
    # copy_rates_from_pos returns newest-first when asking from pos 0
    # Sort by time ascending to be safe
    rates_sorted = sorted(rates, key=lambda r: r["time"])
    closed_rates = rates_sorted[:-1]   # drop the last (forming) bar
    closed_rates = closed_rates[-n:]   # keep at most n

    df = _rates_to_df(np.array(closed_rates, dtype=rates.dtype))
    return df


def get_latest_tick(symbol: str) -> dict:
    """
    Return the most recent tick for *symbol*.

    Returns
    -------
    dict with keys: bid (float), ask (float), time (datetime UTC).

    Raises
    ------
    RuntimeError — if MT5 returns no tick data.
    """
    symbol = symbol.upper()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        err = mt5.last_error()
        log.error("get_latest_tick(%s): no tick data. Last error: %s", symbol, err)
        raise RuntimeError(
            "No tick data for '%s': %s" % (symbol, str(err))
        )
    return {
        "bid":  tick.bid,
        "ask":  tick.ask,
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
    }
