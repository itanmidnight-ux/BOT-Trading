# XAUUSD 1-Bar Scalp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the bot from dual-symbol phased swing trading (EUR_USD/XAU_USD, multi-bar holds) to single-symbol XAUUSD M1 scalping that opens exactly one trade per candle, holds for at most one bar, and uses 1:1 leverage.

**Architecture:** Every M1 bar close forces the previous trade shut (SL/TP if touched intra-bar, otherwise a forced close at that bar's close) and immediately opens the next trade in the ensemble's predicted direction — no HOLD signal, no multi-bar TP2/trailing/partial-close logic. `capital_scaler.py` (the old EUR→XAU phase system) stops being wired into the live pipeline but is left intact as an orphaned, independently-tested module — not deleted.

**Tech Stack:** Python 3.13, pandas, XGBoost/LightGBM/CatBoost ensemble, pytest.

## Global Constraints

- Single symbol: `settings.SYMBOL = "XAUUSD"` (new setting; old `SYMBOL_PHASE1`/`SYMBOL_PHASE2`/`CAPITAL_PHASE2_THRESHOLD`/`LEVERAGE_PHASE1`/`LEVERAGE_PHASE2` stay defined, untouched, for `capital_scaler.py` and its tests only).
- Leverage: `settings.LEVERAGE_XAUUSD = 1` (per explicit user instruction — do not second-guess this in code or comments).
- `ATR_SL_MULTIPLIER = 1.0`, `ATR_TP1_MULTIPLIER = 1.2` (renamed meaning: single TP, no TP2) — values from the Fable 5 research documented in `docs/superpowers/specs/2026-07-13-xauusd-1bar-scalp-design.md`.
- `MAX_SPREAD_USD = 0.35` — the only signal filter that survives; no threshold/regime/MTF gating.
- One open position at a time, exactly one bar of holding time.
- Every task must leave `python -m pytest tests/ -q` passing (see per-task Run commands) before moving to the next task.

---

### Task 1: Settings & constants for single-symbol 1:1 scalping

**Files:**
- Modify: `config/settings.py`
- Modify: `config/constants.py`
- Test: `tests/test_imports.py`

**Interfaces:**
- Produces: `settings.SYMBOL` (str), `settings.LEVERAGE_XAUUSD` (int), `settings.MAX_SPREAD_USD` (float), `settings.ATR_SL_MULTIPLIER` (float, now 1.0), `settings.ATR_TP1_MULTIPLIER` (float, now 1.2), `constants.EXIT_TP` (str), `constants.EXIT_BAR_CLOSE` (str).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_imports.py` (new test function, after `test_config_settings`):

```python
def test_config_settings_scalp():
    from config.settings import SYMBOL, LEVERAGE_XAUUSD, MAX_SPREAD_USD, ATR_SL_MULTIPLIER, ATR_TP1_MULTIPLIER
    assert SYMBOL == "XAUUSD"
    assert LEVERAGE_XAUUSD == 1
    assert MAX_SPREAD_USD == 0.35
    assert ATR_SL_MULTIPLIER == 1.0
    assert ATR_TP1_MULTIPLIER == 1.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_imports.py::test_config_settings_scalp -v`
Expected: FAIL with `ImportError: cannot import name 'SYMBOL'`

- [ ] **Step 3: Update `config/settings.py`**

In the `# ── Instruments ─` block, keep the existing four `SYMBOL_PHASE1`/`SYMBOL_PHASE2`/`CAPITAL_PHASE2_THRESHOLD`/`LEVERAGE_PHASE1`/`LEVERAGE_PHASE2` lines untouched, and add immediately below them:

```python
SYMBOL                     = _get("SYMBOL", "XAUUSD")
LEVERAGE_XAUUSD            = _get("LEVERAGE_XAUUSD", 1)
```

In the `# ── ATR Multipliers ─` block, change the two existing defaults and delete the two now-unused lines:

```python
# ── ATR Multipliers ────────────────────────────────────────────────────────────
ATR_SL_MULTIPLIER       = _get("ATR_SL_MULTIPLIER", 1.0)
ATR_TP1_MULTIPLIER      = _get("ATR_TP1_MULTIPLIER", 1.2)
```

(Delete the `ATR_TP2_MULTIPLIER`, `ATR_TRAILING_MULTIPLIER`, and `PARTIAL_TP_FRACTION` lines that followed — they are no longer referenced anywhere after Task 3.)

In the `# ── Risk Management ─` block, add:

```python
MAX_SPREAD_USD           = _get("MAX_SPREAD_USD", 0.35)
```

- [ ] **Step 4: Update `config/constants.py`**

Replace:

```python
EXIT_SL           = "SL"
EXIT_TP1          = "TP1"
EXIT_TP2          = "TP2"
EXIT_TRAILING     = "TRAILING"
EXIT_TIME         = "TIME_EXIT"
EXIT_ADVERSE      = "ADVERSE"
EXIT_EMERGENCY    = "EMERGENCY"
```

with:

```python
EXIT_SL           = "SL"
EXIT_TP           = "TP"
EXIT_BAR_CLOSE    = "BAR_CLOSE"
EXIT_ADVERSE      = "ADVERSE"
EXIT_EMERGENCY    = "EMERGENCY"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_imports.py -v`
Expected: all PASS, including `test_config_settings_scalp` and the pre-existing `test_config_settings` (still asserts `SYMBOL_PHASE1 == "EURUSD"`, untouched).

- [ ] **Step 6: Commit**

```bash
git add config/settings.py config/constants.py tests/test_imports.py
git commit -m "feat: add single-symbol XAUUSD 1:1 leverage scalp settings"
```

---

### Task 2: Align ML target label with 1-bar holding horizon

**Files:**
- Modify: `core/feature_engine.py:150-176` (the `add_target` static method)
- Test: `tests/test_core_logic.py:48-55` (`TestFeatureEngine::test_target_binary`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `FeatureEngine.add_target(df, symbol)` unchanged signature; `target` column now reflects a 1-candle-ahead move instead of 3-candle-ahead.

- [ ] **Step 1: Write the failing test**

Replace `test_target_binary` in `tests/test_core_logic.py` with:

```python
    def test_target_binary(self):
        from core.feature_engine import FeatureEngine
        fe  = FeatureEngine()
        df  = _make_ohlcv(300)
        out = fe.compute(df)
        out = fe.add_target(out, "XAUUSD")
        assert "target" in out.columns
        assert set(out["target"].dropna().unique()).issubset({0, 1})
        # Horizonte de 1 vela: solo la última fila queda sin target válido.
        assert out["target"].iloc[:-1].notna().all()
        assert out["target"].iloc[-1:].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestFeatureEngine::test_target_binary -v`
Expected: FAIL — last 3 rows are NaN under the current 3-candle-ahead implementation, not just the last row.

- [ ] **Step 3: Update `core/feature_engine.py`**

In `add_target`, replace:

```python
        df["target"] = (
            df["close"].shift(-3) > df["close"] + pip_th
        ).astype(float)

        # The last 3 rows have no valid future close → set to NaN
        df.loc[df.index[-3:], "target"] = np.nan
```

with:

```python
        df["target"] = (
            df["close"].shift(-1) > df["close"] + pip_th
        ).astype(float)

        # The last row has no valid future close → set to NaN
        df.loc[df.index[-1:], "target"] = np.nan
```

Also update the docstring line `"The target is 1 when the close price 3 candles ahead exceeds the"` to `"The target is 1 when the close price 1 candle ahead exceeds the"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestFeatureEngine -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/feature_engine.py tests/test_core_logic.py
git commit -m "fix: align ML target horizon with 1-bar holding period"
```

---

### Task 3: Simplify exit_manager.py to SL / TP / forced bar-close

**Files:**
- Modify: `core/exit_manager.py`
- Test: `tests/test_core_logic.py:88-140` (`TestExitManager`)

**Interfaces:**
- Produces: `OpenPosition(ticket, symbol, direction, lots, entry, sl, tp, open_time, bars_open=0)` (drops `lots_remaining`, `tp1`, `tp2`, `phase`, `trailing_sl`). `ExitManager.evaluate(...)` now only ever returns `ExitAction("CLOSE_FULL", reason)` where reason is `constants.EXIT_SL`, `constants.EXIT_TP`, or `constants.EXIT_BAR_CLOSE`. `ExitManager.calc_levels(...)` returns `{"sl": float, "tp": float}` (no `tp1`/`tp2` keys).
- Consumed by: Task 9 (`live_trader.py`).

- [ ] **Step 1: Write the failing tests**

Replace the `TestExitManager` class in `tests/test_core_logic.py` with:

```python
class TestExitManager:
    def _make_position(self, direction="BUY"):
        from core.exit_manager import OpenPosition
        from datetime import datetime
        return OpenPosition(
            ticket=1, symbol="XAUUSD", direction=direction,
            lots=0.01, entry=2400.00, sl=2399.00, tp=2401.20,
            open_time=datetime.utcnow(), bars_open=0,
        )

    def test_sl_hit_buy(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2399.80, "high": 2399.90, "low": 2398.50, "close": 2398.80, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "SL"

    def test_tp_hit_buy(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2400.50, "high": 2401.50, "low": 2400.40, "close": 2401.30, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "TP"

    def test_forced_bar_close_when_neither_sl_nor_tp_hit(self):
        from core.exit_manager import ExitManager
        em  = ExitManager()
        pos = self._make_position("BUY")
        bar = {"open": 2400.10, "high": 2400.60, "low": 2399.80, "close": 2400.40, "time": 1}
        action = em.evaluate(pos, bar, atr=1.0)
        assert action.action == "CLOSE_FULL"
        assert action.reason == "BAR_CLOSE"

    def test_calc_levels_buy(self):
        from core.exit_manager import ExitManager
        em     = ExitManager()
        levels = em.calc_levels("XAUUSD", "BUY", 2400.00, 1.0)
        assert levels["sl"] < 2400.00
        assert levels["tp"] > 2400.00

    def test_calc_levels_sell(self):
        from core.exit_manager import ExitManager
        em     = ExitManager()
        levels = em.calc_levels("XAUUSD", "SELL", 2400.00, 1.0)
        assert levels["sl"] > 2400.00
        assert levels["tp"] < 2400.00
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestExitManager -v`
Expected: FAIL — `OpenPosition.__init__()` still requires `lots_remaining`, `tp1`, `tp2`, `phase`, `trailing_sl`.

- [ ] **Step 3: Rewrite `core/exit_manager.py`**

Replace the entire file with:

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("exit_manager")


@dataclass
class ExitAction:
    action:    str           # siempre "CLOSE_FULL" en el ciclo de 1 vela
    reason:    str


@dataclass
class OpenPosition:
    ticket:     int
    symbol:     str
    direction:  str
    lots:       float
    entry:      float
    sl:         float
    tp:         float
    open_time:  datetime
    bars_open:  int = 0


class ExitManager:
    """
    Ciclo de 1 vela: cada posición se abre y se cierra dentro del mismo
    bar M1. evaluate() siempre devuelve CLOSE_FULL — por SL, por TP, o
    forzado al cierre de la vela si no se tocó ninguno de los dos.
    """

    def evaluate(self, position: OpenPosition, bar: dict, atr: float) -> ExitAction:
        position.bars_open += 1

        high, low, close = bar["high"], bar["low"], bar["close"]
        is_buy = position.direction == constants.SIGNAL_BUY

        if is_buy and low <= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)
        if not is_buy and high >= position.sl:
            return ExitAction("CLOSE_FULL", constants.EXIT_SL)

        tp_hit = (is_buy and high >= position.tp) or (not is_buy and low <= position.tp)
        if tp_hit:
            return ExitAction("CLOSE_FULL", constants.EXIT_TP)

        return ExitAction("CLOSE_FULL", constants.EXIT_BAR_CLOSE)

    def calc_levels(self, symbol: str, direction: str, entry: float, atr: float) -> dict:
        """Calcula SL y TP para una nueva entrada (hold de 1 vela)."""
        sl_dist = atr * settings.ATR_SL_MULTIPLIER
        tp_dist = atr * settings.ATR_TP1_MULTIPLIER

        if direction == constants.SIGNAL_BUY:
            return {
                "sl": round(entry - sl_dist, 2),
                "tp": round(entry + tp_dist, 2),
            }
        return {
            "sl": round(entry + sl_dist, 2),
            "tp": round(entry - tp_dist, 2),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestExitManager tests/test_imports.py::test_core_exit_manager -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/exit_manager.py tests/test_core_logic.py
git commit -m "feat: simplify exit_manager to SL/TP/forced-bar-close cycle"
```

---

### Task 4: Simplify state_manager position schema

**Files:**
- Modify: `core/state_manager.py:37-54` (`save_position`)
- Test: `tests/test_core_logic.py:142-153` (`TestStateManager::test_save_and_retrieve`)

**Interfaces:**
- Produces: `StateManager.save_position(symbol, ticket, direction, lots, entry, sl, tp, open_time=None)` (drops `tp1`/`tp2` params, replaces with single `tp`; drops `lots_remaining`/`phase`/`trailing_sl` from the persisted dict).
- Consumed by: Task 9 (`live_trader.py`).

- [ ] **Step 1: Write the failing test**

Replace `test_save_and_retrieve` in `tests/test_core_logic.py` with:

```python
    def test_save_and_retrieve(self):
        from core.state_manager import StateManager
        sm = StateManager()
        sm.save_position("XAUUSD", 123, "BUY", 0.01, 2400.00, 2399.00, 2401.20)
        assert sm.has_open_position("XAUUSD")
        pos = sm.get_position("XAUUSD")
        assert pos["ticket"]    == 123
        assert pos["direction"] == "BUY"
        assert pos["tp"]        == 2401.20
        sm.clear_position("XAUUSD")
        assert not sm.has_open_position("XAUUSD")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestStateManager::test_save_and_retrieve -v`
Expected: FAIL — `save_position()` still requires `tp1`/`tp2` positional args and the returned dict has no `tp` key.

- [ ] **Step 3: Update `core/state_manager.py`**

Replace:

```python
    def save_position(self, symbol: str, ticket: int, direction: str,
                      lots: float, entry: float, sl: float, tp1: float, tp2: float):
        self._state["positions"][symbol] = {
            "ticket":          ticket,
            "direction":       direction,
            "lots":            lots,
            "lots_remaining":  lots,
            "entry":           entry,
            "sl":              sl,
            "tp1":             tp1,
            "tp2":             tp2,
            "phase":           1,
            "trailing_sl":     None,
            "bars_open":       0,
            "open_time":       datetime.utcnow().isoformat(),
        }
        self._save()
        _log.debug(f"Estado guardado: {symbol} ticket:{ticket}")
```

with:

```python
    def save_position(self, symbol: str, ticket: int, direction: str,
                      lots: float, entry: float, sl: float, tp: float):
        self._state["positions"][symbol] = {
            "ticket":          ticket,
            "direction":       direction,
            "lots":            lots,
            "entry":           entry,
            "sl":              sl,
            "tp":              tp,
            "bars_open":       0,
            "open_time":       datetime.utcnow().isoformat(),
        }
        self._save()
        _log.debug(f"Estado guardado: {symbol} ticket:{ticket}")
```

Also update `verify_with_mt5` (the `ADOPTED_{symbol}` branch), which currently calls `self.save_position(symbol, mt5_pos.ticket, direction, mt5_pos.volume, mt5_pos.price_open, mt5_pos.sl, mt5_pos.tp, mt5_pos.tp)` — drop the duplicated trailing `mt5_pos.tp` argument:

```python
                self.save_position(symbol, mt5_pos.ticket, direction,
                                   mt5_pos.volume, mt5_pos.price_open,
                                   mt5_pos.sl, mt5_pos.tp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestStateManager -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add core/state_manager.py tests/test_core_logic.py
git commit -m "feat: simplify position state schema to single SL/TP"
```

---

### Task 5: Fix Kelly leverage to use LEVERAGE_XAUUSD

**Files:**
- Modify: `core/kelly.py:137-190` (`fraction_to_lots`, `verify_margin`)

**Interfaces:**
- Produces: `fraction_to_lots`/`verify_margin` now use `settings.LEVERAGE_XAUUSD` unconditionally instead of branching on `settings.LEVERAGE_PHASE1`/`LEVERAGE_PHASE2`.

- [ ] **Step 1: Run existing tests first to confirm current baseline**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestKellyEngine -v`
Expected: all PASS (these tests do not assert on the leverage constant, only on lot bounds — they will keep passing after the change since bounds are clamped by `KELLY_MIN_LOTS`/`KELLY_MAX_LOTS`).

- [ ] **Step 2: Update `core/kelly.py`**

In `fraction_to_lots`, replace:

```python
        leverage = (
            settings.LEVERAGE_PHASE2 if "XAU" in symbol else settings.LEVERAGE_PHASE1
        )
        notional = capital * fraction * leverage
```

with:

```python
        leverage = settings.LEVERAGE_XAUUSD
        notional = capital * fraction * leverage
```

In `verify_margin`, replace:

```python
        leverage = (
            settings.LEVERAGE_PHASE2 if "XAU" in symbol else settings.LEVERAGE_PHASE1
        )
        contract_size = symbol_info.get("trade_contract_size", 100_000)
```

with:

```python
        leverage = settings.LEVERAGE_XAUUSD
        contract_size = symbol_info.get("trade_contract_size", 100_000)
```

- [ ] **Step 3: Run tests to verify they still pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestKellyEngine -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add core/kelly.py
git commit -m "fix: use LEVERAGE_XAUUSD (1:1) for Kelly position sizing"
```

---

### Task 6: Signal generator always emits a direction (no HOLD gating)

**Files:**
- Modify: `core/signal_generator.py`

**Interfaces:**
- Produces: `Signal(symbol, direction, probability, atr, ensemble_votes=0, confidence=False)` (drops `regime`, `mtf_votes`, `threshold_used`, `rl_threshold`). `SignalGenerator.generate(symbol, df_features, feature_cols)` now returns a `Signal` on every call where an ensemble/model probability was computable — never `None` due to threshold/regime/MTF, only `None` when there's no open-position guard trip, insufficient data, or the model itself fails.
- Consumed by: Task 9 (`live_trader.py`, which only reads `signal.direction` and `signal.atr` — unaffected by the dropped fields).

- [ ] **Step 1: Run existing tests first to confirm current baseline**

Run: `source .venv/bin/activate && python -m pytest tests/test_imports.py::test_core_signal_generator -v`
Expected: PASS (this test only checks `SignalGenerator` is callable — it exercises the constructor, not `generate()`, so it is diff-safe).

- [ ] **Step 2: Rewrite `core/signal_generator.py`**

Replace the entire file with:

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("signal_generator")


@dataclass
class Signal:
    symbol:         str
    direction:      str
    probability:    float
    atr:            float
    ensemble_votes: int = 0
    confidence:     bool = False


class SignalGenerator:
    """
    Genera una señal direccional (BUY o SELL) en cada llamada — sin HOLD
    por threshold/régimen/MTF. El único filtro que sobrevive (spread) se
    aplica en live_trader/backtester antes de invocar generate().
    """

    def __init__(self, model_updater, regime_detector, mtf_analyzer, state_manager,
                 ensemble_model=None, lstm_model=None, rl_overlay=None, ollama_advisor=None):
        self._updater  = model_updater
        self._regime   = regime_detector
        self._mtf      = mtf_analyzer
        self._state    = state_manager
        self._ensemble = ensemble_model
        self._lstm     = lstm_model
        self._rl       = rl_overlay
        self._ollama   = ollama_advisor

    def generate(self, symbol: str, df_features: pd.DataFrame,
                 feature_cols: list) -> Optional[Signal]:
        if df_features is None or len(df_features) < 2:
            return None
        if self._state.has_open_position(symbol):
            return None

        atr = float(df_features['atr'].iloc[-1]) if 'atr' in df_features.columns else 0.0
        if atr <= 0:
            return None

        latest = df_features[feature_cols].iloc[-1:].copy().fillna(0)

        # ── Ensemble (principal) ──────────────────────────────────────────────
        proba_buy      = 0.5
        confidence     = False
        ensemble_votes = 0

        if self._ensemble is not None:
            try:
                probas, conf_mask = self._ensemble.predict_with_confidence(latest)
                proba_buy  = float(probas[0])
                confidence = bool(conf_mask[0])
                weights    = self._ensemble.get_weights()
                ensemble_votes = len([w for w in weights.values() if w > 0.25])
            except Exception as e:
                _log.debug(f"Ensemble error: {e}")
                model = self._updater.get_model(symbol)
                if model is not None:
                    try:
                        proba_buy = float(model.predict_proba(latest)[0, 1])
                    except Exception:
                        return None
        else:
            model = self._updater.get_model(symbol)
            if model is None:
                return None
            try:
                proba_buy = float(model.predict_proba(latest)[0, 1])
            except Exception:
                return None

        # ── LSTM voto adicional ───────────────────────────────────────────────
        lstm_proba = 0.5
        if self._lstm is not None:
            try:
                lstm_proba = self._lstm.predict_proba(df_features.tail(25))
            except Exception:
                pass

        if self._lstm is not None and lstm_proba != 0.5:
            proba_buy = 0.70 * proba_buy + 0.30 * lstm_proba

        # ── Dirección — siempre BUY o SELL, nunca HOLD ────────────────────────
        if proba_buy >= 0.5:
            direction = constants.SIGNAL_BUY
            proba     = proba_buy
        else:
            direction = constants.SIGNAL_SELL
            proba     = 1.0 - proba_buy

        _log.info(f"SEÑAL {direction} {symbol} proba:{proba:.3f} conf:{confidence}")

        return Signal(
            symbol=symbol, direction=direction,
            probability=round(proba, 4), atr=round(atr, 6),
            ensemble_votes=ensemble_votes,
            confidence=confidence,
        )
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `source .venv/bin/activate && python -m pytest tests/test_imports.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add core/signal_generator.py
git commit -m "feat: signal generator always emits a direction, drop HOLD gating"
```

---

### Task 7: Backtester — chained one-trade-per-candle simulation

**Files:**
- Modify: `core/backtester.py`
- Test: `tests/test_core_logic.py:170-219` (`TestBacktester`)

**Interfaces:**
- Consumes: `settings.MAX_SPREAD_USD`, `settings.LEVERAGE_XAUUSD`, `settings.ATR_SL_MULTIPLIER`, `settings.ATR_TP1_MULTIPLIER` (Task 1), `constants.EXIT_SL`/`EXIT_TP`/`EXIT_BAR_CLOSE` (Task 1).
- Produces: `Backtester.run(df, model, feature_cols, signal_threshold, initial_capital, regime_detector=None, mtf_analyzer=None) -> dict` — **signature unchanged** (`signal_threshold`/`regime_detector`/`mtf_analyzer` accepted for call-site compatibility with `training_loop.py`/`analysis/optimizer.py`, but no longer used to filter trades — every bar without an open position opens one, subject only to the spread filter). Every trade opened is closed on the very next bar (SL, TP, or forced at that bar's close) — no multi-bar holds.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core_logic.py`, inside `TestBacktester`:

```python
    def test_one_bar_max_hold(self):
        """Cada trade se cierra en la vela siguiente a la entrada, nunca más tarde."""
        from core.backtester import Backtester
        from core.feature_engine import FeatureEngine
        from unittest.mock import MagicMock
        import numpy as np

        fe   = FeatureEngine()
        df   = _make_ohlcv(500)
        df_f = fe.compute(df)
        df_f = fe.add_target(df_f, "XAUUSD")
        df_f = df_f.dropna().reset_index(drop=True)
        feature_cols = fe.get_feature_cols()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        bt      = Backtester("XAUUSD")
        metrics = bt.run(df_f, mock_model, feature_cols, 0.62, 20.0)

        trades_df = metrics["trades_df"]
        assert len(trades_df) > 0
        assert (trades_df["bars_open"] == 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestBacktester::test_one_bar_max_hold -v`
Expected: FAIL — current implementation gates on `signal_threshold`/regime and allows multi-bar holds (`bars_open` can exceed 1).

- [ ] **Step 3: Rewrite `core/backtester.py`**

Replace the entire file with:

```python
"""
Backtester vela a vela estilo MT5.
Ciclo de 1 vela: cada trade se abre y se cierra en la vela siguiente a la
señal — sin TP2, sin trailing, sin hold multi-vela. Simula spread y slippage.
"""
import random
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger
from config import settings, constants

_log = get_logger("backtester")

random.seed(42)


class Backtester:
    def __init__(self, symbol: str):
        self.symbol   = symbol
        self._results: list = []

    def run(self, df: pd.DataFrame, model, feature_cols: list,
            signal_threshold: float, initial_capital: float,
            regime_detector=None, mtf_analyzer=None) -> dict:
        """
        Simula trading vela a vela sobre df (set de test). Un trade por vela:
        se fuerza el cierre del trade abierto y de inmediato se evalúa/abre
        el siguiente, sin huecos. signal_threshold/regime_detector/mtf_analyzer
        se aceptan por compatibilidad de firma pero ya no filtran señales —
        el único filtro es el spread simulado vs MAX_SPREAD_USD.
        Retorna dict con métricas completas.
        """
        self._results = []
        capital       = initial_capital
        open_trade    = None
        equity_curve  = [capital]

        is_xau   = "XAU" in self.symbol
        spread   = (settings.SPREAD_POINTS_XAUUSD if is_xau else settings.SPREAD_POINTS_EURUSD)
        point    = 0.01 if is_xau else 0.00001
        spread_f = spread * point

        for i in range(len(df)):
            row  = df.iloc[i]
            bar  = {"open": row["open"], "high": row["high"],
                    "low": row["low"],   "close": row["close"],
                    "time": row.get("time", i)}
            atr  = float(row["atr"]) if "atr" in row else spread_f * 15

            # ── Cierre forzado del trade abierto en la vela anterior ─────────
            if open_trade is not None:
                open_trade["bars_open"] += 1
                pnl, reason, exit_price = self._check_exit(open_trade, bar, atr)
                capital += pnl
                self._record_trade(open_trade, exit_price, pnl, reason, capital)
                equity_curve.append(capital)
                open_trade = None

            if i + 1 >= len(df):
                break

            # ── Único filtro: spread por encima del máximo tolerado ──────────
            if spread_f > settings.MAX_SPREAD_USD:
                continue

            try:
                x = df[feature_cols].iloc[i:i+1].fillna(0)
                proba_buy = float(model.predict_proba(x)[0, 1])
            except Exception:
                continue

            direction, proba = self._get_direction(proba_buy)

            next_bar = df.iloc[i + 1]
            slippage = random.uniform(0, 0.5) * point
            entry    = next_bar["open"] + (slippage if direction == constants.SIGNAL_BUY
                                           else -slippage)
            if direction == constants.SIGNAL_BUY:
                entry += spread_f

            lots = self._calc_lots(capital, atr)
            if lots <= 0:
                continue

            sl = entry - atr * settings.ATR_SL_MULTIPLIER  if direction == constants.SIGNAL_BUY \
                 else entry + atr * settings.ATR_SL_MULTIPLIER
            tp = entry + atr * settings.ATR_TP1_MULTIPLIER if direction == constants.SIGNAL_BUY \
                 else entry - atr * settings.ATR_TP1_MULTIPLIER

            pip_value = self._pip_value(lots, self.symbol)

            open_trade = {
                "entry":      entry,
                "direction":  direction,
                "lots":       lots,
                "sl":         sl,
                "tp":         tp,
                "bars_open":  0,
                "open_time":  row.get("time", i),
                "pip_value":  pip_value,
                "atr_entry":  atr,
                "proba":      proba,
            }

        if open_trade is not None:
            last = df.iloc[-1]
            pnl  = self._calc_pnl(open_trade, last["close"])
            capital += pnl
            self._record_trade(open_trade, last["close"], pnl, constants.EXIT_BAR_CLOSE, capital)
            equity_curve.append(capital)

        return self._compute_metrics(initial_capital, capital, equity_curve)

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(self, trade: dict, bar: dict, atr: float):
        """Siempre devuelve un resultado: SL, TP, o cierre forzado al close de la vela."""
        is_buy = trade["direction"] == constants.SIGNAL_BUY
        high, low, close = bar["high"], bar["low"], bar["close"]

        if is_buy and low <= trade["sl"]:
            return self._calc_pnl(trade, trade["sl"]), constants.EXIT_SL, trade["sl"]
        if not is_buy and high >= trade["sl"]:
            return self._calc_pnl(trade, trade["sl"]), constants.EXIT_SL, trade["sl"]

        tp_hit = (is_buy and high >= trade["tp"]) or (not is_buy and low <= trade["tp"])
        if tp_hit:
            return self._calc_pnl(trade, trade["tp"]), constants.EXIT_TP, trade["tp"]

        return self._calc_pnl(trade, close), constants.EXIT_BAR_CLOSE, close

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_direction(proba_buy: float):
        if proba_buy >= 0.5:
            return constants.SIGNAL_BUY, proba_buy
        return constants.SIGNAL_SELL, 1.0 - proba_buy

    @staticmethod
    def _pip_value(lots: float, symbol: str) -> float:
        if "XAU" in symbol:
            return lots * 100 * 0.1   # 1 lot = 100oz, pip = $0.1/oz
        return lots * 100000 * 0.0001  # 1 lot = 100k EUR, pip = $10

    def _calc_pnl(self, trade: dict, exit_price: float) -> float:
        diff = exit_price - trade["entry"] if trade["direction"] == constants.SIGNAL_BUY \
               else trade["entry"] - exit_price
        pip_div = 0.1 if "XAU" in self.symbol else 0.00010
        pips    = diff / pip_div
        return round(pips * trade["pip_value"], 4)

    def _calc_lots(self, capital: float, atr: float) -> float:
        fraction = settings.KELLY_BOOTSTRAP_FRAC
        leverage = settings.LEVERAGE_XAUUSD
        notional = capital * fraction * leverage
        if "XAU" in self.symbol:
            lots = notional / (2300 * 100)
        else:
            lots = notional / 100000
        lots = round(lots, 2)
        return max(settings.KELLY_MIN_LOTS, min(settings.KELLY_MAX_LOTS, lots))

    def _record_trade(self, trade: dict, exit_price: float, pnl: float, reason: str, capital: float):
        pip_div = 0.1 if "XAU" in self.symbol else 0.00010
        diff    = exit_price - trade["entry"] if trade["direction"] == constants.SIGNAL_BUY \
                  else trade["entry"] - exit_price
        pips    = round(diff / pip_div, 1)
        self._results.append({
            "open_time":   trade["open_time"],
            "symbol":      self.symbol,
            "direction":   trade["direction"],
            "lots":        trade["lots"],
            "entry":       trade["entry"],
            "exit_price":  exit_price,
            "sl":          trade["sl"],
            "tp":          trade["tp"],
            "pips":        pips,
            "pnl_usd":     pnl,
            "capital":     round(capital, 4),
            "reason":      reason,
            "bars_open":   trade["bars_open"],
            "proba":       trade.get("proba", 0),
        })

    def _compute_metrics(self, initial_capital: float, final_capital: float,
                          equity_curve: list) -> dict:
        if not self._results:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "net_pnl": 0, "final_capital": final_capital,
                    "max_drawdown": 0, "sharpe": 0, "roi": 0,
                    "equity_curve": equity_curve}

        df    = pd.DataFrame(self._results)
        wins  = df[df["pnl_usd"] > 0]
        loses = df[df["pnl_usd"] <= 0]

        win_rate      = len(wins) / len(df) if len(df) > 0 else 0
        sum_wins      = wins["pnl_usd"].sum()
        sum_losses    = loses["pnl_usd"].abs().sum()
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else float("inf")
        net_pnl       = final_capital - initial_capital
        roi           = net_pnl / initial_capital

        returns = df["pnl_usd"].values
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0

        peak    = initial_capital
        max_dd  = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        self._save_csv(df)

        return {
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(profit_factor, 3),
            "total_trades":  len(df),
            "wins":          len(wins),
            "losses":        len(loses),
            "net_pnl":       round(net_pnl, 4),
            "final_capital": round(final_capital, 4),
            "max_drawdown":  round(max_dd, 4),
            "sharpe":        round(float(sharpe), 3),
            "roi":           round(roi, 4),
            "avg_win_pips":  round(wins["pips"].mean(), 2) if len(wins) > 0 else 0,
            "avg_loss_pips": round(loses["pips"].abs().mean(), 2) if len(loses) > 0 else 0,
            "equity_curve":  equity_curve,
            "trades_df":     df,
        }

    def _save_csv(self, df: pd.DataFrame):
        path = settings.LOGS_BACKTEST_DIR / \
               f"backtest_{self.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
        settings.LOGS_BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["trades_df"] if "trades_df" in df.columns else [], errors="ignore") \
          .to_csv(path, index=False)
        _log.info(f"Backtest guardado: {path.name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_core_logic.py::TestBacktester -v`
Expected: all PASS, including the new `test_one_bar_max_hold`.

- [ ] **Step 5: Commit**

```bash
git add core/backtester.py tests/test_core_logic.py
git commit -m "feat: backtester simulates chained one-trade-per-candle cycle"
```

---

### Task 8: Training loop tunes SL/TP only (drop dead threshold tuning)

**Files:**
- Modify: `core/training_loop.py`

**Interfaces:**
- Consumes: `Backtester.run(...)` (Task 7, signature unchanged).
- Produces: `TrainingLoop` public interface (`__init__`, `.run(df_raw, initial_capital, max_iterations, verbose)`) unchanged; internal `_adjust_params`/`_apply_temp_params`/`_restore_params`/`_save_optimized_params` no longer touch `SIGNAL_THRESHOLD` (dead lever now that `signal_generator`/`backtester` don't gate on it — see Tasks 6-7).

- [ ] **Step 1: Run existing tests first to confirm current baseline**

Run: `source .venv/bin/activate && python -m pytest tests/test_imports.py::test_core_training_loop -v`
Expected: PASS (constructor-only smoke test, diff-safe).

- [ ] **Step 2: Update `core/training_loop.py`**

Replace the `__init__` body's parameter-tracking lines:

```python
        self._threshold   = settings.SIGNAL_THRESHOLD
        self._sl_mult     = settings.ATR_SL_MULTIPLIER
        self._tp1_mult    = settings.ATR_TP1_MULTIPLIER
```

with:

```python
        self._sl_mult     = settings.ATR_SL_MULTIPLIER
        self._tp1_mult    = settings.ATR_TP1_MULTIPLIER
```

Replace the log line in `run()`:

```python
            _log.info(f"  threshold={self._threshold:.3f} sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")
```

with:

```python
            _log.info(f"  sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")
```

Replace the backtest call (drop `signal_threshold=self._threshold`, keep the rest):

```python
                bt_metrics = bt.run(
                    df       = df_test,
                    model    = model,
                    feature_cols   = feature_cols,
                    signal_threshold = self._threshold,
                    initial_capital  = initial_capital,
                    regime_detector  = self._regime,
                )
```

with:

```python
                bt_metrics = bt.run(
                    df       = df_test,
                    model    = model,
                    feature_cols   = feature_cols,
                    signal_threshold = settings.SIGNAL_THRESHOLD,
                    initial_capital  = initial_capital,
                    regime_detector  = self._regime,
                )
```

Replace the Ollama suggestion block:

```python
            if self._ollama is not None and self._ollama.is_available() and iteration % 3 == 0:
                try:
                    suggestion = self._ollama.analyze_performance(
                        {"win_rate": win_rate, "profit_factor": pf, "trades": n_trades,
                         "threshold": self._threshold}, self.symbol)
                    if suggestion and suggestion.get("param") == "SIGNAL_THRESHOLD":
                        adj = float(suggestion.get("amount", 0.02))
                        if suggestion.get("action") == "increase":
                            self._threshold = min(0.75, self._threshold + adj)
                        elif suggestion.get("action") == "decrease":
                            self._threshold = max(0.55, self._threshold - adj)
                        _log.info(f"  Ollama: {suggestion.get('reason', '')} → thr={self._threshold:.3f}")
                except Exception:
                    pass
```

with:

```python
            if self._ollama is not None and self._ollama.is_available() and iteration % 3 == 0:
                try:
                    suggestion = self._ollama.analyze_performance(
                        {"win_rate": win_rate, "profit_factor": pf, "trades": n_trades,
                         "sl_mult": self._sl_mult, "tp1_mult": self._tp1_mult}, self.symbol)
                    if suggestion and suggestion.get("param") == "ATR_SL_MULTIPLIER":
                        adj = float(suggestion.get("amount", 0.1))
                        if suggestion.get("action") == "increase":
                            self._sl_mult = min(2.5, self._sl_mult + adj)
                        elif suggestion.get("action") == "decrease":
                            self._sl_mult = max(0.5, self._sl_mult - adj)
                        _log.info(f"  Ollama: {suggestion.get('reason', '')} → sl_mult={self._sl_mult:.2f}")
                except Exception:
                    pass
```

Replace `_adjust_params` entirely:

```python
    def _adjust_params(self, iteration: int, win_rate: float, profit_factor: float):
        """
        Estrategia de ajuste progresivo para maximizar WR sin overfitting.
        """
        deficit = settings.MIN_WIN_RATE_LIVE - win_rate

        if iteration <= 5:
            # Fase 1: ajustar threshold para filtrar señales débiles
            if win_rate < 0.50:
                self._threshold = min(0.75, self._threshold + 0.03)
            elif win_rate < 0.55:
                self._threshold = min(0.72, self._threshold + 0.02)
            else:
                self._threshold = min(0.70, self._threshold + 0.01)

        elif iteration <= 10:
            # Fase 2: dar más espacio al SL para reducir stop-outs prematuros
            if profit_factor < 1.3:
                self._sl_mult = min(2.5, self._sl_mult + 0.15)
            # También ajusta threshold
            if win_rate < 0.55:
                self._threshold = min(0.72, self._threshold + 0.015)

        elif iteration <= 15:
            # Fase 3: aumentar TP para mejorar profit factor
            if profit_factor < 1.5:
                self._tp1_mult = min(3.0, self._tp1_mult + 0.1)
            # Reducir threshold si ya filtramos demasiado
            if win_rate < 0.50 and self._threshold > 0.65:
                self._threshold = max(0.60, self._threshold - 0.01)

        else:
            # Fase 4: ajuste combinado agresivo
            if win_rate < 0.50:
                self._sl_mult     = min(2.8, self._sl_mult + 0.2)
                self._threshold   = min(0.75, self._threshold + 0.02)
                self._tp1_mult    = min(3.5, self._tp1_mult + 0.15)
            elif win_rate < 0.55:
                self._threshold   = min(0.73, self._threshold + 0.01)
                self._sl_mult     = min(2.5, self._sl_mult + 0.1)

        _log.debug(f"  Ajuste: threshold={self._threshold:.3f} "
                   f"sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")
```

with:

```python
    def _adjust_params(self, iteration: int, win_rate: float, profit_factor: float):
        """
        Ajusta SL/TP para converger al gate de WR — con hold de 1 vela, el
        threshold ya no filtra señales, así que SL/TP son las únicas
        palancas que cambian el resultado del backtest.
        """
        if win_rate < 0.50:
            # Muchos stop-outs: da más espacio al SL y acorta el TP para
            # asegurar ganancias chicas más seguido.
            self._sl_mult  = min(2.0, self._sl_mult + 0.10)
            self._tp1_mult = max(0.5, self._tp1_mult - 0.05)
        elif win_rate < settings.MIN_WIN_RATE_LIVE:
            self._sl_mult = min(1.8, self._sl_mult + 0.05)
        elif profit_factor < 1.1:
            # WR ya alcanza el gate pero el profit factor es débil: agranda el TP.
            self._tp1_mult = min(2.0, self._tp1_mult + 0.10)

        _log.debug(f"  Ajuste: sl_mult={self._sl_mult:.2f} tp1_mult={self._tp1_mult:.2f}")
```

Replace `_apply_temp_params`/`_restore_params`:

```python
    def _apply_temp_params(self):
        """Aplica parámetros temporales al settings module."""
        self._orig_threshold = settings.SIGNAL_THRESHOLD
        self._orig_sl        = settings.ATR_SL_MULTIPLIER
        self._orig_tp1       = settings.ATR_TP1_MULTIPLIER
        settings.SIGNAL_THRESHOLD   = self._threshold
        settings.ATR_SL_MULTIPLIER  = self._sl_mult
        settings.ATR_TP1_MULTIPLIER = self._tp1_mult

    def _restore_params(self):
        """Restaura parámetros originales."""
        settings.SIGNAL_THRESHOLD   = self._orig_threshold
        settings.ATR_SL_MULTIPLIER  = self._orig_sl
        settings.ATR_TP1_MULTIPLIER = self._orig_tp1
```

with:

```python
    def _apply_temp_params(self):
        """Aplica parámetros temporales al settings module."""
        self._orig_sl  = settings.ATR_SL_MULTIPLIER
        self._orig_tp1 = settings.ATR_TP1_MULTIPLIER
        settings.ATR_SL_MULTIPLIER  = self._sl_mult
        settings.ATR_TP1_MULTIPLIER = self._tp1_mult

    def _restore_params(self):
        """Restaura parámetros originales."""
        settings.ATR_SL_MULTIPLIER  = self._orig_sl
        settings.ATR_TP1_MULTIPLIER = self._orig_tp1
```

Replace `_save_optimized_params`:

```python
    def _save_optimized_params(self):
        """Guarda parámetros optimizados en runtime_params.json."""
        path = settings.CONFIG_DIR / "runtime_params.json"
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update({
                "SIGNAL_THRESHOLD":   round(self._threshold, 3),
                "ATR_SL_MULTIPLIER":  round(self._sl_mult, 2),
                "ATR_TP1_MULTIPLIER": round(self._tp1_mult, 2),
            })
            path.write_text(json.dumps(current, indent=2))
            _log.info(f"Parámetros optimizados guardados: thr={self._threshold:.3f} "
                      f"sl={self._sl_mult:.2f} tp1={self._tp1_mult:.2f}")
        except Exception as e:
            _log.warning(f"No se pudo guardar runtime_params: {e}")
```

with:

```python
    def _save_optimized_params(self):
        """Guarda parámetros optimizados en runtime_params.json."""
        path = settings.CONFIG_DIR / "runtime_params.json"
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update({
                "ATR_SL_MULTIPLIER":  round(self._sl_mult, 2),
                "ATR_TP1_MULTIPLIER": round(self._tp1_mult, 2),
            })
            path.write_text(json.dumps(current, indent=2))
            _log.info(f"Parámetros optimizados guardados: sl={self._sl_mult:.2f} tp1={self._tp1_mult:.2f}")
        except Exception as e:
            _log.warning(f"No se pudo guardar runtime_params: {e}")
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: all PASS (same set as after Task 7 — this task only touches `training_loop.py` internals, no test directly exercises `.run()` end-to-end at unit level besides `test_full_pipeline.py`, which is covered in Task 11).

- [ ] **Step 4: Commit**

```bash
git add core/training_loop.py
git commit -m "refactor: training loop tunes SL/TP only, drop dead threshold tuning"
```

---

### Task 9: Live trader — 1-bar forced-close cycle, drop capital_scaler wiring

**Files:**
- Modify: `core/live_trader.py`

**Interfaces:**
- Consumes: `OpenPosition`/`ExitAction` (Task 3), `StateManager.save_position(symbol, ticket, direction, lots, entry, sl, tp)` (Task 4), `ExitManager.calc_levels(...) -> {"sl", "tp"}` (Task 3).
- Produces: `LiveTrader.__init__(symbol, mt5_connector, mt5_stream, feature_engine, model_updater, regime_detector, mtf_analyzer, signal_generator, kelly_engine, risk_manager, trade_manager, exit_manager, state_manager, data_updater, auto_improver)` — **drops the `capital_scaler` parameter**.
- Consumed by: Task 10 (`main.py`).

- [ ] **Step 1: Run existing tests first to confirm current baseline**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: same pass/fail set as end of Task 8 (no test directly instantiates `LiveTrader`, so this is a behavior-only change with no direct unit test — verified instead by Task 11's full-suite + smoke run).

- [ ] **Step 2: Update `core/live_trader.py` constructor**

Replace:

```python
    def __init__(self, symbol: str, mt5_connector, mt5_stream,
                 feature_engine, model_updater, regime_detector,
                 mtf_analyzer, signal_generator, kelly_engine,
                 risk_manager, trade_manager, exit_manager,
                 state_manager, data_updater, capital_scaler,
                 auto_improver):
        self.symbol         = symbol
        self._conn          = mt5_connector
        self._stream        = mt5_stream
        self._fe            = feature_engine
        self._model_upd     = model_updater
        self._regime        = regime_detector
        self._mtf           = mtf_analyzer
        self._signal        = signal_generator
        self._kelly         = kelly_engine
        self._risk          = risk_manager
        self._trades        = trade_manager
        self._exit          = exit_manager
        self._state         = state_manager
        self._data_upd      = data_updater
        self._capital_scaler = capital_scaler
        self._improver      = auto_improver
```

with:

```python
    def __init__(self, symbol: str, mt5_connector, mt5_stream,
                 feature_engine, model_updater, regime_detector,
                 mtf_analyzer, signal_generator, kelly_engine,
                 risk_manager, trade_manager, exit_manager,
                 state_manager, data_updater,
                 auto_improver):
        self.symbol         = symbol
        self._conn          = mt5_connector
        self._stream        = mt5_stream
        self._fe            = feature_engine
        self._model_upd     = model_updater
        self._regime        = regime_detector
        self._mtf           = mtf_analyzer
        self._signal        = signal_generator
        self._kelly         = kelly_engine
        self._risk          = risk_manager
        self._trades        = trade_manager
        self._exit          = exit_manager
        self._state         = state_manager
        self._data_upd      = data_updater
        self._improver      = auto_improver
```

- [ ] **Step 3: Remove the phase-transition block from `_on_bar_close`**

Delete this block (step 7 of the numbered comments):

```python
            # 7. Chequeo de fase
            old_capital = self._state.capital
            transition  = self._capital_scaler.check_transition(old_capital, equity)
            if transition:
                notifier.notify(notifier.PHASE_CHANGE, {
                    "phase":   transition["phase"],
                    "capital": equity,
                    "symbol":  transition.get("symbol", self.symbol),
                })
                self._state.phase = transition["phase"]

            # 8. Nueva señal (si no hay posición abierta)
```

Replace with (renumber the following comment only, logic unchanged):

```python
            # 7. Nueva señal (si no hay posición abierta)
```

- [ ] **Step 4: Simplify `_manage_open_position`**

Replace the whole method:

```python
    def _manage_open_position(self, bar: dict, atr: float, equity: float):
        """Evalúa y gestiona la posición abierta."""
        pos_data = self._state.get_position(self.symbol)
        if pos_data is None:
            return

        from core.exit_manager import OpenPosition
        position = OpenPosition(
            ticket     = pos_data["ticket"],
            symbol     = self.symbol,
            direction  = pos_data["direction"],
            lots       = pos_data.get("lots_remaining", pos_data["lots"]),
            lots_remaining = pos_data.get("lots_remaining", pos_data["lots"]),
            entry      = pos_data["entry"],
            sl         = pos_data["sl"],
            tp1        = pos_data["tp1"],
            tp2        = pos_data["tp2"],
            phase      = pos_data.get("phase", 1),
            trailing_sl = pos_data.get("trailing_sl"),
            open_time  = datetime.fromisoformat(pos_data["open_time"]) if isinstance(pos_data["open_time"], str) else pos_data["open_time"],
            bars_open  = pos_data.get("bars_open", 0),
        )

        action = self._exit.evaluate(position, bar, atr)

        if action.action == "CLOSE_FULL":
            success = self._trades.close_position(
                position.ticket, self.symbol, position.direction
            )
            if success:
                # Calcula PnL estimado
                tick    = self._stream.get_latest_tick(self.symbol)
                cur_p   = tick.get("bid" if position.direction == constants.SIGNAL_BUY else "ask", position.entry) if tick else position.entry
                diff    = cur_p - position.entry if position.direction == constants.SIGNAL_BUY else position.entry - cur_p
                pips    = diff / (0.00010 if "XAU" not in self.symbol else 0.1)
                pnl     = pips * position.lots * (10 if "XAU" not in self.symbol else 100)

                win = pnl > 0
                self._pnl_today += pnl
                self._trades_today += 1
                if win:
                    self._wins_today += 1

                self._state.record_trade_result(win)
                self._state.capital = max(0, self._state.capital + pnl)
                self._state.clear_position(self.symbol)
                self._kelly.update(self.symbol, win, abs(pips))

                notifier.notify(notifier.TRADE_CLOSE, {
                    "symbol":    self.symbol,
                    "direction": position.direction,
                    "pips":      round(pips, 1),
                    "pnl":       round(pnl, 4),
                    "reason":    action.reason,
                })

        elif action.action == "CLOSE_PARTIAL":
            half_lots = round(position.lots * settings.PARTIAL_TP_FRACTION, 2)
            half_lots = max(half_lots, settings.KELLY_MIN_LOTS)
            self._trades.close_position(position.ticket, self.symbol,
                                        position.direction, lots=half_lots)
            # Actualiza estado: fase 2, nuevo SL, trailing
            self._state.update_position(self.symbol,
                                         phase=2,
                                         sl=action.new_sl,
                                         trailing_sl=action.new_sl,
                                         lots_remaining=position.lots - half_lots,
                                         bars_open=position.bars_open)
            self._trades.modify_position_sl(position.ticket, self.symbol, action.new_sl)

        elif action.action == "MOVE_SL":
            self._trades.modify_position_sl(position.ticket, self.symbol, action.new_sl)
            self._state.update_position(self.symbol,
                                         sl=action.new_sl,
                                         trailing_sl=action.new_sl,
                                         bars_open=position.bars_open)

        elif action.action == "HOLD":
            self._state.update_position(self.symbol, bars_open=position.bars_open)
```

with:

```python
    def _manage_open_position(self, bar: dict, atr: float, equity: float):
        """
        Evalúa la posición abierta. Ciclo de 1 vela: evaluate() siempre
        devuelve CLOSE_FULL (por SL, TP, o forzado al cierre de la vela).
        """
        pos_data = self._state.get_position(self.symbol)
        if pos_data is None:
            return

        from core.exit_manager import OpenPosition
        position = OpenPosition(
            ticket     = pos_data["ticket"],
            symbol     = self.symbol,
            direction  = pos_data["direction"],
            lots       = pos_data["lots"],
            entry      = pos_data["entry"],
            sl         = pos_data["sl"],
            tp         = pos_data["tp"],
            open_time  = datetime.fromisoformat(pos_data["open_time"]) if isinstance(pos_data["open_time"], str) else pos_data["open_time"],
            bars_open  = pos_data.get("bars_open", 0),
        )

        action = self._exit.evaluate(position, bar, atr)

        success = self._trades.close_position(
            position.ticket, self.symbol, position.direction
        )
        if success:
            tick    = self._stream.get_latest_tick(self.symbol)
            cur_p   = tick.get("bid" if position.direction == constants.SIGNAL_BUY else "ask", position.entry) if tick else position.entry
            diff    = cur_p - position.entry if position.direction == constants.SIGNAL_BUY else position.entry - cur_p
            pips    = diff / 0.1
            pnl     = pips * position.lots * 100

            win = pnl > 0
            self._pnl_today += pnl
            self._trades_today += 1
            if win:
                self._wins_today += 1

            self._state.record_trade_result(win)
            self._state.capital = max(0, self._state.capital + pnl)
            self._state.clear_position(self.symbol)
            self._kelly.update(self.symbol, win, abs(pips))

            notifier.notify(notifier.TRADE_CLOSE, {
                "symbol":    self.symbol,
                "direction": position.direction,
                "pips":      round(pips, 1),
                "pnl":       round(pnl, 4),
                "reason":    action.reason,
            })
```

(Note: `pips`/`pnl` are hardcoded to the XAUUSD formula since `settings.SYMBOL` is XAUUSD-only now — matches the same formula already used unconditionally in `_check_exit`/`_calc_pnl` in `backtester.py` for the XAU branch.)

- [ ] **Step 5: Update `_execute_signal` to use single `tp` level**

Replace:

```python
    def _execute_signal(self, signal, equity: float, free_margin: float):
        """Ejecuta una señal: calcula lots, verifica margen, coloca orden."""
        levels = self._exit.calc_levels(self.symbol, signal.direction, 0, signal.atr)

        # Precio actual como entrada estimada
        tick   = self._stream.get_latest_tick(self.symbol)
        if tick is None:
            return
        entry  = tick.get("ask" if signal.direction == constants.SIGNAL_BUY else "bid", 0)
        if entry <= 0:
            return

        # Recalcula niveles con entry real
        levels = self._exit.calc_levels(self.symbol, signal.direction, entry, signal.atr)

        # Kelly position sizing
        self._kelly.recalculate(self.symbol)
        fraction = self._kelly.get_current_fraction(self.symbol)
        symbol_info = self._conn.get_symbol_info(self.symbol)
        lots = self._kelly.fraction_to_lots(fraction, equity, self.symbol, entry)

        if not self._kelly.verify_margin(lots, self.symbol, free_margin, symbol_info):
            _log.warning(f"Margen insuficiente para {lots} lots — reduciendo a mínimo")
            lots = settings.KELLY_MIN_LOTS

        # Enviar orden
        result = self._trades.open_market_order(
            symbol    = self.symbol,
            direction = signal.direction,
            lots      = lots,
            sl_price  = levels["sl"],
            tp_price  = levels["tp1"],
        )

        if result.success:
            self._state.save_position(
                symbol    = self.symbol,
                ticket    = result.ticket,
                direction = signal.direction,
                lots      = lots,
                entry     = result.entry,
                sl        = levels["sl"],
                tp1       = levels["tp1"],
                tp2       = levels["tp2"],
            )
            notifier.notify(notifier.TRADE_OPEN, {
                "symbol":    self.symbol,
                "direction": signal.direction,
                "lots":      lots,
                "entry":     result.entry,
                "sl":        levels["sl"],
                "tp1":       levels["tp1"],
            })
        else:
            _log.warning(f"Orden fallida: {result.message}")
```

with:

```python
    def _execute_signal(self, signal, equity: float, free_margin: float):
        """Ejecuta una señal: calcula lots, verifica margen, coloca orden."""
        # Precio actual como entrada estimada
        tick   = self._stream.get_latest_tick(self.symbol)
        if tick is None:
            return
        entry  = tick.get("ask" if signal.direction == constants.SIGNAL_BUY else "bid", 0)
        if entry <= 0:
            return

        levels = self._exit.calc_levels(self.symbol, signal.direction, entry, signal.atr)

        # Kelly position sizing
        self._kelly.recalculate(self.symbol)
        fraction = self._kelly.get_current_fraction(self.symbol)
        symbol_info = self._conn.get_symbol_info(self.symbol)
        lots = self._kelly.fraction_to_lots(fraction, equity, self.symbol, entry)

        if not self._kelly.verify_margin(lots, self.symbol, free_margin, symbol_info):
            _log.warning(f"Margen insuficiente para {lots} lots — reduciendo a mínimo")
            lots = settings.KELLY_MIN_LOTS

        result = self._trades.open_market_order(
            symbol    = self.symbol,
            direction = signal.direction,
            lots      = lots,
            sl_price  = levels["sl"],
            tp_price  = levels["tp"],
        )

        if result.success:
            self._state.save_position(
                symbol    = self.symbol,
                ticket    = result.ticket,
                direction = signal.direction,
                lots      = lots,
                entry     = result.entry,
                sl        = levels["sl"],
                tp        = levels["tp"],
            )
            notifier.notify(notifier.TRADE_OPEN, {
                "symbol":    self.symbol,
                "direction": signal.direction,
                "lots":      lots,
                "entry":     result.entry,
                "sl":        levels["sl"],
                "tp":        levels["tp"],
            })
        else:
            _log.warning(f"Orden fallida: {result.message}")
```

- [ ] **Step 6: Run tests to verify nothing broke**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: same pass/fail set as end of Task 8.

- [ ] **Step 7: Commit**

```bash
git add core/live_trader.py
git commit -m "feat: live_trader forces 1-bar close cycle, drop capital_scaler wiring"
```

---

### Task 10: main.py — single symbol, drop CapitalScaler wiring

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `settings.SYMBOL` (Task 1), `LiveTrader.__init__(...)` without `capital_scaler` (Task 9).

- [ ] **Step 1: Run existing tests first to confirm current baseline**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: same pass/fail set as end of Task 9.

- [ ] **Step 2: Update `main.py`**

Replace every `settings.SYMBOL_PHASE1` occurrence with `settings.SYMBOL`:

- `run_backtest_only(settings.SYMBOL_PHASE1)` → `run_backtest_only(settings.SYMBOL)` (2 call sites: the `_TEST_MODE` branch and menu option `[3]`)
- `run_live(settings.SYMBOL_PHASE1, "demo")` → `run_live(settings.SYMBOL, "demo")`
- `run_live(settings.SYMBOL_PHASE1, "real")` → `run_live(settings.SYMBOL, "real")`
- `run_optimize(settings.SYMBOL_PHASE1)` → `run_optimize(settings.SYMBOL)`

Replace the log-analysis loop:

```python
def run_log_analysis():
    """Analiza todos los logs y muestra resumen."""
    from analysis.log_analyzer import LogAnalyzer
    analyzer = LogAnalyzer()
    for symbol in [settings.SYMBOL_PHASE1, settings.SYMBOL_PHASE2]:
        result = analyzer.analyze(symbol)
```

with:

```python
def run_log_analysis():
    """Analiza todos los logs y muestra resumen."""
    from analysis.log_analyzer import LogAnalyzer
    analyzer = LogAnalyzer()
    for symbol in [settings.SYMBOL]:
        result = analyzer.analyze(symbol)
```

In `run_live`, remove the `CapitalScaler` import and instantiation, and drop `capital_scaler=scaler,` from the `LiveTrader(...)` call:

```python
    from core.state_manager import StateManager
    from core.auto_improver import AutoImprover
    from core.live_trader import LiveTrader
    from core.ollama_advisor import OllamaAdvisor
    from core.objective_engine import ObjectiveEngine
    from analysis.performance_reporter import PerformanceReporter
```

(This import block already excludes `CapitalScaler` — locate and delete the separate line `from core.capital_scaler import CapitalScaler` that currently sits among the other imports at the top of `run_live`.)

Replace:

```python
    signal_gen = SignalGenerator(model_upd, regime, mtf, state)

    live = LiveTrader(
        symbol=symbol, mt5_connector=connector, mt5_stream=stream,
        feature_engine=fe, model_updater=model_upd, regime_detector=regime,
        mtf_analyzer=mtf, signal_generator=signal_gen, kelly_engine=kelly,
        risk_manager=risk, trade_manager=trade_mgr, exit_manager=exit_mgr,
        state_manager=state, data_updater=data_upd, capital_scaler=scaler,
        auto_improver=improver,
    )
```

with:

```python
    signal_gen = SignalGenerator(model_upd, regime, mtf, state)

    live = LiveTrader(
        symbol=symbol, mt5_connector=connector, mt5_stream=stream,
        feature_engine=fe, model_updater=model_upd, regime_detector=regime,
        mtf_analyzer=mtf, signal_generator=signal_gen, kelly_engine=kelly,
        risk_manager=risk, trade_manager=trade_mgr, exit_manager=exit_mgr,
        state_manager=state, data_updater=data_upd,
        auto_improver=improver,
    )
```

Also remove the now-unused `scaler = CapitalScaler()` line and the dashboard log line that referenced `scaler.get_phase(equity)`:

```python
    print(f"  Balance: ${equity:.2f} | Fase: {scaler.get_phase(equity)} | WR modelo: {result['win_rate']:.1%}")
```

with:

```python
    print(f"  Balance: ${equity:.2f} | WR modelo: {result['win_rate']:.1%}")
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`
Expected: same pass/fail set as end of Task 9.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: main.py trades single XAUUSD symbol, drop CapitalScaler wiring"
```

---

### Task 11: Full-suite verification + manual smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: same 9 pre-existing failures as documented in the prior Linux-cleanup session (all `ModuleNotFoundError: pandas_ta` — unrelated, fix pending user's `sudo pip install pandas-ta`), zero *new* failures introduced by this plan. If `pandas_ta` has since been installed, expect all tests PASS.

- [ ] **Step 2: Manual smoke test — offline backtest mode**

Run: `source .venv/bin/activate && python main.py --test`
Expected: banner prints, `[TEST MODE]` runs `run_backtest_only(settings.SYMBOL)` (now XAUUSD) against yfinance data (ticker resolves to `"XAUUSD=X"` per the existing `"XAU" in symbol` branch in `main.py:run_backtest_only`), training loop runs up to `MAX_TRAIN_ITERS`, and prints a final WR/PF/trades summary without raising.

- [ ] **Step 3: Confirm trade frequency matches "1 trade per candle" intent**

Run: `source .venv/bin/activate && python -c "
import pandas as pd, glob
files = sorted(glob.glob('logs/backtest/backtest_XAUUSD_*.csv'))
assert files, 'no backtest CSV found — run Step 2 first'
df = pd.read_csv(files[-1])
print('trades:', len(df))
print('bars_open value_counts:'); print(df['bars_open'].value_counts())
"`
Expected: `bars_open` column is `1` for every row (confirms the one-candle hold cycle end-to-end, not just at the unit-test level).

- [ ] **Step 4: Report results to user**

No commit — this task is verification-only. Summarize pass/fail counts and the smoke-test WR/PF/trade-count for the user.

---

## Self-Review Notes

- **Spec coverage:** symbol/leverage (Task 1, 5), 1-bar forced cycle (Task 3, 7, 9), always-directional signal (Task 6), ATR SL/TP values from Fable research (Task 1), spread-only filter (Task 7), capital_scaler deprecated-not-deleted (Task 9-10, verified untouched in `core/capital_scaler.py` and its tests), ML target horizon alignment (Task 2, not in original spec but required for correctness — added during planning). All covered.
- **Known deviation from spec doc:** the spec's `ATR_TP_MULTIPLIER` name is kept as the pre-existing `ATR_TP1_MULTIPLIER` to avoid an unnecessary rename cascade into `analysis/optimizer.py`; the *value* (1.2) matches the spec exactly.
- **Type/signature consistency:** `OpenPosition`/`ExitAction`/`Signal`/`StateManager.save_position` field names verified consistent across Tasks 3, 4, 6, 9. `Backtester.run()` signature deliberately left unchanged (Task 7) to avoid touching `analysis/optimizer.py`, which is out of scope for this plan.
