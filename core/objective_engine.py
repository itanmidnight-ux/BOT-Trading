"""
Autonomous objective system that sets and tracks improvement goals.
Uses a predefined ladder of increasingly ambitious trading targets;
beyond the top rung, an optional OllamaAdvisor generates new objectives.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from utils.logger import get_logger
from config import settings

_log = get_logger("objective_engine")

OBJECTIVES_FILE = settings.LOGS_SYSTEM_DIR / "objectives.json"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Objective:
    id: int
    name: str
    target_wr: float          # minimum win-rate to achieve
    target_pf: float          # minimum profit-factor to achieve
    target_trades: int        # minimum trades before evaluation
    max_drawdown: float       # maximum allowed drawdown fraction
    status: str = "active"    # active | achieved | failed
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    achieved_at: Optional[str] = None
    notes: str = ""


# ── Predefined ladder ─────────────────────────────────────────────────────────

OBJECTIVE_LADDER: List[Objective] = [
    Objective(1, "Bootstrap",  0.59, 1.5,   50,  0.15),
    Objective(2, "Stable",     0.62, 1.7,  100,  0.13),
    Objective(3, "Proficient", 0.65, 2.0,  150,  0.12),
    Objective(4, "Advanced",   0.68, 2.2,  200,  0.10),
    Objective(5, "Expert",     0.70, 2.5,  300,  0.08),
    Objective(6, "Master",     0.73, 3.0,  500,  0.06),
    Objective(7, "Elite",      0.75, 3.5, 1000,  0.05),
]


# ── Engine ────────────────────────────────────────────────────────────────────

class ObjectiveEngine:
    """
    Tracks the bot's current improvement objective, evaluates progress,
    and advances through the ladder automatically.

    Args:
        ollama_advisor: optional OllamaAdvisor instance used to generate
                        custom objectives once the predefined ladder is
                        exhausted.
    """

    def __init__(self, ollama_advisor=None):
        self._advisor = ollama_advisor
        # Deep-copy the ladder so mutations don't affect the module constant
        self._ladder: List[Objective] = [
            Objective(**asdict(o)) for o in OBJECTIVE_LADDER
        ]
        self._current_idx: int = 0          # index into self._ladder
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current(self) -> Objective:
        """Return the currently active objective."""
        return self._ladder[self._current_idx]

    def evaluate(
        self,
        win_rate: float,
        profit_factor: float,
        max_dd: float,
        n_trades: int,
    ) -> dict:
        """
        Evaluate current performance against the active objective.

        Returns:
            {'achieved': True, 'next': Objective}  if objective met, or
            {'achieved': False, 'failed': True, 'next': Objective}  if failed, or
            {'achieved': False, 'progress': {wr_pct, pf_pct, overall_pct}}
        """
        obj = self.get_current()

        # Not enough trades yet → return progress only
        if n_trades < obj.target_trades:
            return {
                "achieved": False,
                "progress": self.get_progress(win_rate, profit_factor, n_trades),
            }

        # Check achievement criteria
        wr_ok = win_rate >= obj.target_wr
        pf_ok = profit_factor >= obj.target_pf
        dd_ok = max_dd <= obj.max_drawdown

        if wr_ok and pf_ok and dd_ok:
            obj.status      = "achieved"
            obj.achieved_at = datetime.utcnow().isoformat()
            obj.notes       = (
                f"WR={win_rate:.1%} PF={profit_factor:.2f} "
                f"DD={max_dd:.1%} after {n_trades} trades"
            )
            _log.info(f"Objective [{obj.id}] {obj.name!r} ACHIEVED. Advancing …")
            self.advance()
            self.save()
            return {"achieved": True, "next": self.get_current()}

        # Failure check: twice the required trades and still missing both metrics
        overdue  = n_trades > obj.target_trades * 2
        both_bad = not wr_ok and not pf_ok
        if overdue and both_bad:
            obj.status = "failed"
            obj.notes  = (
                f"Failed after {n_trades} trades: "
                f"WR={win_rate:.1%} (need {obj.target_wr:.1%}), "
                f"PF={profit_factor:.2f} (need {obj.target_pf:.2f})"
            )
            _log.warning(f"Objective [{obj.id}] {obj.name!r} FAILED. Adjusting target …")
            self._downgrade_current(win_rate, profit_factor)
            self.save()
            return {
                "achieved": False,
                "failed":   True,
                "next":     self.get_current(),
            }

        return {
            "achieved": False,
            "progress": self.get_progress(win_rate, profit_factor, n_trades),
        }

    def get_progress(
        self,
        win_rate: float,
        profit_factor: float,
        n_trades: int,
    ) -> dict:
        """
        Return fractional progress (0–1) toward each metric of the current
        objective, plus an overall weighted score.
        """
        obj = self.get_current()

        # WR progress: 0 at start (50% baseline), 1 at target
        wr_baseline = 0.50
        wr_range    = max(obj.target_wr - wr_baseline, 1e-9)
        wr_pct      = min(max((win_rate - wr_baseline) / wr_range, 0.0), 1.0)

        # PF progress: 1.0 baseline
        pf_baseline = 1.0
        pf_range    = max(obj.target_pf - pf_baseline, 1e-9)
        pf_pct      = min(max((profit_factor - pf_baseline) / pf_range, 0.0), 1.0)

        # Trade count progress toward minimum
        trade_pct = min(n_trades / max(obj.target_trades, 1), 1.0)

        overall = round((wr_pct * 0.45 + pf_pct * 0.40 + trade_pct * 0.15), 4)

        return {
            "wr_pct":     round(wr_pct, 4),
            "pf_pct":     round(pf_pct, 4),
            "trade_pct":  round(trade_pct, 4),
            "overall_pct": overall,
        }

    def advance(self) -> None:
        """
        Move to the next objective in the ladder.
        If the ladder is exhausted and an OllamaAdvisor is available,
        generate a custom objective; otherwise repeat the last rung.
        """
        next_idx = self._current_idx + 1

        if next_idx < len(self._ladder):
            # Ensure the next rung is reset to active status
            nxt = self._ladder[next_idx]
            if nxt.status != "active":
                nxt.status      = "active"
                nxt.achieved_at = None
            self._current_idx = next_idx
            _log.info(
                f"Advanced to objective [{nxt.id}] {nxt.name!r}: "
                f"WR≥{nxt.target_wr:.0%} PF≥{nxt.target_pf:.2f}"
            )
            return

        # Beyond the predefined ladder
        _log.info("All predefined objectives completed — generating custom objective.")
        custom = self._generate_custom_objective()
        self._ladder.append(custom)
        self._current_idx = len(self._ladder) - 1

    def save(self) -> None:
        """Persist the full ladder state (including progress) to disk."""
        try:
            OBJECTIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "current_idx": self._current_idx,
                "ladder": [asdict(o) for o in self._ladder],
                "saved_at": datetime.utcnow().isoformat(),
            }
            OBJECTIVES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            _log.warning(f"Failed to save objectives: {exc}")

    def display(
        self,
        win_rate: float = 0.0,
        profit_factor: float = 0.0,
        n_trades: int = 0,
    ) -> str:
        """
        Return a compact one-line status string, e.g.:

            [OBJ 2/7] Stable | Target: WR≥62% PF≥1.7 | Current: WR=61% PF=1.8 | Progress: 95%
        """
        obj     = self.get_current()
        total   = len(self._ladder)
        prog    = self.get_progress(win_rate, profit_factor, n_trades)
        overall = int(prog["overall_pct"] * 100)

        return (
            f"[OBJ {obj.id}/{total}] {obj.name} | "
            f"Target: WR≥{obj.target_wr:.0%} PF≥{obj.target_pf:.1f} | "
            f"Current: WR={win_rate:.0%} PF={profit_factor:.1f} | "
            f"Progress: {overall}%"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted state from disk if available."""
        if not OBJECTIVES_FILE.exists():
            return
        try:
            data = json.loads(OBJECTIVES_FILE.read_text(encoding="utf-8"))
            saved_ladder = data.get("ladder", [])
            for i, obj_data in enumerate(saved_ladder):
                if i < len(self._ladder):
                    # Restore mutable fields only; keep type hints intact
                    for key in ("status", "achieved_at", "notes", "created_at"):
                        if key in obj_data:
                            setattr(self._ladder[i], key, obj_data[key])
                else:
                    # Custom objective appended beyond predefined ladder
                    try:
                        self._ladder.append(Objective(**obj_data))
                    except TypeError:
                        pass
            saved_idx = data.get("current_idx", 0)
            if 0 <= saved_idx < len(self._ladder):
                self._current_idx = saved_idx
            _log.debug(f"Objectives loaded from {OBJECTIVES_FILE}")
        except Exception as exc:
            _log.warning(f"Could not load objectives file: {exc}")

    def _downgrade_current(
        self, actual_wr: float, actual_pf: float
    ) -> None:
        """
        Replace the failed objective with an easier one based on actual metrics,
        giving the bot a reachable near-term target.
        """
        obj         = self.get_current()
        new_target_wr = round(min(actual_wr + 0.02, obj.target_wr - 0.01), 2)
        new_target_pf = round(min(actual_pf + 0.10, obj.target_pf - 0.05), 2)
        replacement  = Objective(
            id            = obj.id,
            name          = f"{obj.name}-Revised",
            target_wr     = new_target_wr,
            target_pf     = new_target_pf,
            target_trades = int(obj.target_trades * 0.75),
            max_drawdown  = round(obj.max_drawdown + 0.02, 2),
            status        = "active",
            notes         = f"Downgraded from original {obj.name} target.",
        )
        _log.info(
            f"Downgraded objective: WR {obj.target_wr:.0%}→{new_target_wr:.0%} "
            f"PF {obj.target_pf:.2f}→{new_target_pf:.2f}"
        )
        self._ladder[self._current_idx] = replacement

    def _generate_custom_objective(self) -> Objective:
        """
        Generate an objective beyond the predefined ladder.
        Uses OllamaAdvisor if available, otherwise applies a fixed increment.
        """
        last = self._ladder[-1]
        new_id = last.id + 1

        if self._advisor is not None and self._advisor.is_available():
            suggestion = self._advisor.generate_objective(
                current_wr=last.target_wr,
                current_pf=last.target_pf,
            )
            if suggestion:
                target_wr      = float(suggestion.get("target_wr", last.target_wr + 0.01))
                target_pf      = float(suggestion.get("target_pf", last.target_pf + 0.1))
                strategy_note  = str(suggestion.get("strategy", "LLM-generated target"))
                timeframe_t    = int(suggestion.get("timeframe_trades", last.target_trades + 200))
                return Objective(
                    id            = new_id,
                    name          = f"Custom-{new_id}",
                    target_wr     = round(min(target_wr, 0.95), 2),
                    target_pf     = round(min(target_pf, 10.0), 2),
                    target_trades = timeframe_t,
                    max_drawdown  = max(last.max_drawdown - 0.01, 0.03),
                    notes         = strategy_note,
                )

        # Fallback: fixed increment
        return Objective(
            id            = new_id,
            name          = f"Beyond-Elite-{new_id}",
            target_wr     = round(min(last.target_wr + 0.01, 0.95), 2),
            target_pf     = round(min(last.target_pf + 0.25, 10.0), 2),
            target_trades = last.target_trades + 500,
            max_drawdown  = max(last.max_drawdown - 0.01, 0.03),
            notes         = "Auto-incremented beyond predefined ladder.",
        )
