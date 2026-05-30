"""
rl_overlay.py

Q-Learning agent that dynamically adjusts the signal probability threshold
based on discretized market state (regime, volatility bin, session hour,
win-streak).  The learned Q-table is persisted to disk so that knowledge
survives bot restarts.

State space  : 4 regimes × 3 vol bins × 6 hour bins × 3 streak bins = 216
Action space : 3  (−0.02 / 0.00 / +0.02 adjustment to base threshold)
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from config import settings

_log = get_logger("rl_overlay")

RL_FILE: Path = settings.LOGS_SYSTEM_DIR / "rl_state.json"

# ── Action definitions ────────────────────────────────────────────────────────
_ADJUSTMENTS = [-0.02, 0.0, 0.02]   # action 0 / 1 / 2

# ── Threshold safety bounds ───────────────────────────────────────────────────
_THR_MIN = 0.52
_THR_MAX = 0.80

# ── Regime encoding ───────────────────────────────────────────────────────────
_REGIME_MAP: dict[str, int] = {
    "TRENDING_UP":   0,
    "TRENDING_DOWN": 1,
    "RANGING":       2,
    "VOLATILE":      3,
}


class RLOverlay:
    """
    Epsilon-greedy Q-Learning overlay that adjusts ``settings.SIGNAL_THRESHOLD``
    by ±0.02 (or keeps it) depending on the current market state.

    Usage
    -----
    ::

        rl = RLOverlay()

        # Before evaluating a signal:
        threshold = rl.get_threshold(regime, atr, atr_mean, hour, win_streak)

        # After a trade closes:
        reward = rl.record_trade(pnl_usd, pips_gained)
        rl.update(reward, next_regime, next_atr, next_atr_mean,
                  next_hour, next_streak)
    """

    N_STATES  = 216   # 4 × 3 × 6 × 3
    N_ACTIONS = 3

    def __init__(self) -> None:
        self.q_table     = np.zeros((self.N_STATES, self.N_ACTIONS))
        self.epsilon     = 0.15    # ε-greedy exploration rate
        self.alpha       = 0.10    # learning rate
        self.gamma       = 0.90    # discount factor
        self.base_thr    = float(settings.SIGNAL_THRESHOLD)
        self.current_thr = self.base_thr

        self._last_state:  Optional[int] = None
        self._last_action: Optional[int] = None

        RL_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_threshold(
        self,
        regime:     str,
        atr:        float,
        atr_mean:   float,
        hour:       int,
        win_streak: int,
    ) -> float:
        """
        Return the adjusted signal threshold for the current market state.

        Parameters
        ----------
        regime     : one of TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE
        atr        : current ATR value
        atr_mean   : mean ATR over recent look-back window (e.g. 50 bars)
        hour       : UTC hour of the current bar (0-23)
        win_streak : number of consecutive winning trades (0+)

        Returns
        -------
        float  threshold in [_THR_MIN, _THR_MAX]
        """
        state = self._encode_state(regime, atr, atr_mean, hour, win_streak)

        if np.random.random() < self.epsilon:
            action = int(np.random.randint(self.N_ACTIONS))
        else:
            action = int(np.argmax(self.q_table[state]))

        self._last_state  = state
        self._last_action = action

        self.current_thr = float(
            np.clip(self.base_thr + _ADJUSTMENTS[action], _THR_MIN, _THR_MAX)
        )

        _log.debug(
            "RL threshold: state=%d action=%d (%+.2f) → thr=%.4f ε=%.4f",
            state, action, _ADJUSTMENTS[action], self.current_thr, self.epsilon,
        )
        return self.current_thr

    def update(
        self,
        reward:        float,
        next_regime:   str,
        next_atr:      float,
        next_atr_mean: float,
        next_hour:     int,
        next_streak:   int,
    ) -> None:
        """
        Apply a Q-Learning update using the observed reward and the next state.

        Should be called immediately after ``record_trade``.

        Parameters
        ----------
        reward        : scalar reward (use ``record_trade`` to compute it)
        next_*        : market state *after* the trade closed
        """
        if self._last_state is None or self._last_action is None:
            return

        next_state = self._encode_state(
            next_regime, next_atr, next_atr_mean, next_hour, next_streak
        )

        current_q  = self.q_table[self._last_state, self._last_action]
        max_next_q = float(np.max(self.q_table[next_state]))
        td_error   = reward + self.gamma * max_next_q - current_q
        new_q      = current_q + self.alpha * td_error

        self.q_table[self._last_state, self._last_action] = new_q

        # Slow epsilon decay — keeps a minimum exploration floor
        self.epsilon = max(0.05, self.epsilon * 0.9995)

        _log.debug(
            "RL update: s=%d a=%d r=%.4f td=%.4f new_q=%.4f ε=%.4f",
            self._last_state, self._last_action,
            reward, td_error, new_q, self.epsilon,
        )

        self._save()

    @staticmethod
    def record_trade(pnl: float, pips: float = 0.0) -> float:
        """
        Convert trade outcome to a bounded reward signal.

        Uses ``tanh`` so that:
          - A $2 win  → reward ≈ +0.76
          - A $0 wash → reward =  0.00
          - A $2 loss → reward ≈ −0.76
          - Extreme values are clipped towards ±1

        Parameters
        ----------
        pnl  : trade profit/loss in USD (positive = win, negative = loss)
        pips : trade pip gain (currently unused, available for extension)

        Returns
        -------
        float  reward in (−1, +1)
        """
        return float(np.tanh(pnl / 2.0))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_state(
        regime:     str,
        atr:        float,
        atr_mean:   float,
        hour:       int,
        win_streak: int,
    ) -> int:
        """
        Discretize continuous market state into a single integer index.

        Encoding: state = regime×54 + vol_bin×18 + hour_bin×3 + streak_bin

        - regime     : 0-3  (4 values)
        - vol_bin    : 0-2  low / mid / high ATR vs mean
        - hour_bin   : 0-5  (4-hour UTC blocks)
        - streak_bin : 0-2  capped at 2 (streak of 3+ → 2)

        Total: 4 × 3 × 6 × 3 = 216 unique states
        """
        r = _REGIME_MAP.get(regime, 2)  # unknown → RANGING

        # Volatility bin
        if atr_mean > 0:
            ratio = atr / atr_mean
        else:
            ratio = 1.0
        if ratio < 0.8:
            v = 0    # low
        elif ratio < 1.5:
            v = 1    # mid
        else:
            v = 2    # high

        h = max(0, min(5, hour // 4))    # 4-hour blocks 0-5
        s = min(2, max(0, win_streak))   # cap at 2

        return r * 54 + v * 18 + h * 3 + s

    def _save(self) -> None:
        """Persist Q-table and epsilon to disk (atomic write)."""
        payload = {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
        }
        tmp = RL_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(RL_FILE)
            _log.debug("RL state saved → %s", RL_FILE)
        except Exception as exc:
            _log.warning("RL save failed: %s", exc)

    def _load(self) -> None:
        """Restore Q-table and epsilon from disk if a checkpoint exists."""
        if not RL_FILE.exists():
            _log.info("No RL checkpoint found — starting fresh.")
            return
        try:
            data = json.loads(RL_FILE.read_text())
            q_list = data.get("q_table")
            if q_list is not None:
                loaded = np.array(q_list, dtype=np.float64)
                if loaded.shape == (self.N_STATES, self.N_ACTIONS):
                    self.q_table = loaded
                    _log.info("RL Q-table restored from %s", RL_FILE)
                else:
                    _log.warning(
                        "RL checkpoint shape mismatch %s — reinitialising.",
                        loaded.shape,
                    )
            eps = data.get("epsilon")
            if eps is not None:
                self.epsilon = float(np.clip(eps, 0.05, 1.0))
        except Exception as exc:
            _log.warning("RL load failed (%s) — starting fresh.", exc)
