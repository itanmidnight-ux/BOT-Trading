"""
mt5_connector.py
Manages the connection lifecycle with a MetaTrader 5 terminal.

Design notes:
- initialize() first tries mt5.initialize() against the already-running terminal.
- If that fails it spawns the terminal directly and polls for up to 30 s:
    the terminal only exists as a Windows binary, so it is launched via
    `wine terminal64.exe` inside WINEPREFIX, using the virtual DISPLAY
    configured in .env (typically an Xvfb display).
- mt5.login() called only when no active session found.
- Exponential back-off: attempts at 0 s, 2 s, 6 s (3 attempts total).
- Every public method guards against a missing terminal with is_connected().
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from core.mt5_compat import mt5
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

log = get_logger("mt5_connector")

# Retry policy
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = [0, 2, 6]          # sleep before attempt 0, 1, 2

# Polling policy when spawning MT5
_POLL_INTERVAL = 2.0                   # seconds between mt5.initialize() polls
_LAUNCH_TIMEOUT = 30.0                 # max seconds to wait after spawning

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_mt5_exe() -> str:
    """Return the default MT5 terminal path (terminal64.exe under Wine)."""
    wineprefix = os.environ.get("WINEPREFIX", str(Path.home() / ".wine_mt5"))
    return str(Path(wineprefix) / "drive_c" / "Program Files" / "MetaTrader 5" / "terminal64.exe")


class MT5Connector:
    """
    Owns the connection lifecycle to a single MT5 terminal instance.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_initialize_once(self) -> bool:
        """
        Attempt mt5.initialize() and if needed mt5.login() from .env credentials.
        Returns True on success, False otherwise.
        """
        # Phase 1: direct connect (session already active in terminal)
        if mt5.initialize():
            if mt5.account_info() is not None:
                log.debug("mt5.initialize() succeeded, account_info available.")
                return True
            # Terminal connected but no active session — try login from .env
            login    = os.environ.get("MT5_LOGIN")
            password = os.environ.get("MT5_PASSWORD")
            server   = os.environ.get("MT5_SERVER")
            if login and password and server:
                log.info("No active session — logging in: %s @ %s", login, server)
                if mt5.login(int(login), password=password, server=server):
                    if mt5.account_info() is not None:
                        log.info("mt5.login() succeeded.")
                        return True
                log.warning("mt5.login() failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

        # Phase 2: initialize with credentials directly
        login    = os.environ.get("MT5_LOGIN")
        password = os.environ.get("MT5_PASSWORD")
        server   = os.environ.get("MT5_SERVER")
        if login and password and server:
            if mt5.initialize(login=int(login), password=password, server=server):
                if mt5.account_info() is not None:
                    log.info("mt5.initialize(credentials) succeeded.")
                    return True
                mt5.shutdown()

        log.debug("mt5.initialize() failed. Last error: %s", mt5.last_error())
        return False

    def _spawn_terminal(self) -> Optional[subprocess.Popen]:
        """Launch the MT5 terminal via Wine."""
        mt5_exe = os.environ.get("MT5_EXE", "") or _default_mt5_exe()
        if not Path(mt5_exe).exists():
            log.warning("MT5_EXE not found: %s", mt5_exe)
            return None

        cmd = ["wine", mt5_exe]
        log.info("Launching MT5 terminal: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("MT5 process spawned (PID %d).", proc.pid)
            return proc
        except FileNotFoundError:
            log.error(
                "No se encontró el ejecutable '%s'. En Linux instala Wine "
                "(ver README) y verifica MT5_EXE / WINEPREFIX en .env.",
                cmd[0],
            )
            return None
        except Exception as exc:
            log.error("Error spawning MT5: %s", exc)
            return None

    def _poll_until_ready(self, timeout: float = _LAUNCH_TIMEOUT) -> bool:
        """
        Repeatedly call mt5.initialize() every _POLL_INTERVAL seconds until the
        terminal is ready or *timeout* seconds have elapsed.
        """
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            log.debug("Poll attempt %d — waiting for MT5 terminal …", attempt)
            if self._try_initialize_once():
                log.info("MT5 terminal became available.")
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_POLL_INTERVAL, remaining))
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Connect to the MT5 terminal.

        Strategy:
          1. Try mt5.initialize() directly (terminal already running).
          2. If that fails, spawn the terminal and poll for up to 30 s.
          3. Apply exponential back-off across up to _MAX_ATTEMPTS total rounds.

        Returns:
            True once connected, False if all attempts failed.
        """
        for attempt_idx in range(_MAX_ATTEMPTS):
            sleep_s = _BACKOFF_SECONDS[attempt_idx]
            if sleep_s > 0:
                log.info(
                    "Back-off: waiting %d s before attempt %d/%d …",
                    sleep_s, attempt_idx + 1, _MAX_ATTEMPTS,
                )
                time.sleep(sleep_s)

            log.info("Connection attempt %d/%d …", attempt_idx + 1, _MAX_ATTEMPTS)

            # --- Phase 1: direct initialize (terminal already running) ----------
            if self._try_initialize_once():
                info = mt5.account_info()
                log.info(
                    "Connected to MT5. Login: %s | Server: %s | Balance: %.2f %s",
                    info.login, info.server, info.balance, info.currency,
                )
                return True

            # --- Phase 2: spawn terminal and poll --------------------------------
            log.warning("Direct mt5.initialize() failed — attempting to spawn terminal.")
            self._proc = self._spawn_terminal()
            if self._proc is None:
                log.error("Cannot spawn MT5 terminal (see warnings above).")
                continue  # try next back-off round

            if self._poll_until_ready(_LAUNCH_TIMEOUT):
                info = mt5.account_info()
                log.info(
                    "Connected to MT5 (after launch). Login: %s | Server: %s | Balance: %.2f %s",
                    info.login, info.server, info.balance, info.currency,
                )
                return True

            log.error(
                "MT5 terminal did not become ready within %.0f s (attempt %d/%d).",
                _LAUNCH_TIMEOUT, attempt_idx + 1, _MAX_ATTEMPTS,
            )

        log.error(
            "Could not connect to MetaTrader 5 after %d attempts. "
            "Ensure the terminal is running and account_info is available.",
            _MAX_ATTEMPTS,
        )
        return False

    def is_connected(self) -> bool:
        """
        Return True when MT5 is initialised and account_info is accessible.
        This is a lightweight check — it does NOT re-initialise.
        """
        try:
            return mt5.account_info() is not None
        except Exception:
            return False

    def get_account_info(self) -> dict:
        """
        Return key account metrics as a plain dict.

        Returns:
            dict with keys: login, server, currency, balance, equity,
                            margin, free_margin.

        Raises:
            RuntimeError: if not connected or account_info returns None.
        """
        info = mt5.account_info()
        if info is None:
            err = mt5.last_error()
            log.error("get_account_info(): mt5.account_info() returned None. Error: %s", err)
            raise RuntimeError("MT5 not connected or account_info unavailable: %s" % str(err))

        return {
            "login":       info.login,
            "server":      info.server,
            "currency":    info.currency,
            "balance":     info.balance,
            "equity":      info.equity,
            "margin":      info.margin,
            "free_margin": info.margin_free,
        }

    def get_symbol_info(self, symbol: str) -> dict:
        """
        Return trading specification for *symbol*.

        Returns:
            dict with keys: point, volume_min, volume_step,
                            trade_contract_size, spread_float.

        Raises:
            RuntimeError: if the symbol is unknown or MT5 is not connected.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            err = mt5.last_error()
            log.error(
                "get_symbol_info(%s): symbol not found or MT5 not connected. Error: %s",
                symbol, err,
            )
            raise RuntimeError("Symbol '%s' not found in MT5: %s" % (symbol, str(err)))

        return {
            "point":               info.point,
            "volume_min":          info.volume_min,
            "volume_step":         info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "spread_float":        bool(info.spread_float),
        }

    def shutdown(self) -> None:
        """
        Release the MT5 Python binding without closing the terminal application.
        Safe to call even if not initialised.
        """
        try:
            mt5.shutdown()
            log.info("MT5 Python binding released (terminal remains open).")
        except Exception as exc:
            log.warning("Exception during mt5.shutdown(): %s", exc)
