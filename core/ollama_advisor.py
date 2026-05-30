"""
Ollama LLM integration for autonomous bot improvement decisions.
Uses local phi3:mini (fallback: gemma2:2b) for parameter suggestions,
strategy recommendations, and trade explanations.
"""
import json
import requests
from pathlib import Path
from datetime import datetime
from utils.logger import get_logger
from config import settings

_log = get_logger("ollama_advisor")

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_BASE = "http://localhost:11434/"
MODEL       = "phi3:mini"   # fallback: gemma2:2b
_DECISIONS_DIR = settings.LOGS_SYSTEM_DIR


class OllamaAdvisor:
    """Wraps a local Ollama instance to guide runtime bot decisions."""

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and responding."""
        try:
            r = requests.get(OLLAMA_BASE, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── Core LLM call ─────────────────────────────────────────────────────────

    def _ask(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Send *prompt* to the local Ollama model.
        Returns the response text or "" on any error.
        """
        payload = {
            "model":   MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
            },
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except requests.exceptions.Timeout:
            _log.warning("Ollama request timed out after 30 s")
            return ""
        except Exception as exc:
            _log.warning(f"Ollama _ask error: {exc}")
            return ""

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_performance(self, metrics: dict, symbol: str) -> dict:
        """
        Given a performance-metrics dict, ask the LLM which single parameter
        to adjust and how.

        Expected *metrics* keys (all optional but helpful):
            win_rate, profit_factor, max_drawdown, n_trades

        Returns dict with keys: param, action, amount, reason
        or {} on parse failure.
        """
        prompt = (
            f"Given these trading metrics: {json.dumps(metrics)}, "
            "what single parameter should I adjust to improve win rate? "
            "Respond ONLY with JSON: "
            '{\"param\": \"SIGNAL_THRESHOLD\", \"action\": \"increase\", '
            '\"amount\": 0.02, \"reason\": \"brief reason\"}'
        )
        raw = self._ask(prompt, max_tokens=150)
        result = self._parse_json(raw)
        if result:
            self._save_decision("analyze_performance", symbol, metrics, result)
        else:
            _log.debug(f"analyze_performance: could not parse JSON from: {raw!r}")
        return result

    def suggest_strategy(
        self, regime: str, win_rate: float, recent_pnl: float
    ) -> str:
        """
        Return a one-sentence strategy recommendation (~10 words).

        Args:
            regime:     e.g. "trending", "ranging", "volatile"
            win_rate:   float 0-1
            recent_pnl: recent realised P&L in USD
        """
        prompt = (
            f"Market regime: {regime}. "
            f"Bot WR: {win_rate:.0%}. "
            f"Recent PnL: ${recent_pnl:.2f}. "
            "In 1 sentence, should the bot be more aggressive or conservative? "
            "Answer in exactly 10 words."
        )
        response = self._ask(prompt, max_tokens=60)
        self._save_decision(
            "suggest_strategy",
            symbol="",
            context={"regime": regime, "win_rate": win_rate, "recent_pnl": recent_pnl},
            result={"suggestion": response},
        )
        return response

    def generate_objective(
        self, current_wr: float, current_pf: float
    ) -> dict:
        """
        Ask the LLM to generate the next improvement objective beyond the
        predefined ladder.

        Returns dict with keys: target_wr, target_pf, strategy, timeframe_trades
        or {} on parse failure.
        """
        prompt = (
            f"Trading bot achieved WR={current_wr:.0%} PF={current_pf:.2f}. "
            "Generate next improvement objective as JSON: "
            '{\"target_wr\": 0.XX, \"target_pf\": X.X, '
            '\"strategy\": \"brief description\", \"timeframe_trades\": 100}'
        )
        raw = self._ask(prompt, max_tokens=150)
        result = self._parse_json(raw)
        if result:
            self._save_decision(
                "generate_objective",
                symbol="",
                context={"current_wr": current_wr, "current_pf": current_pf},
                result=result,
            )
        else:
            _log.debug(f"generate_objective: could not parse JSON from: {raw!r}")
        return result

    def explain_trade(self, signal_dict: dict, regime: str) -> str:
        """
        Return a brief one-line explanation of why the bot is entering a trade.
        Suitable for terminal display.

        Args:
            signal_dict: signal data (direction, confidence, features …)
            regime:      current market regime label
        """
        direction  = signal_dict.get("direction", "BUY")
        confidence = signal_dict.get("confidence", signal_dict.get("probability", 0.0))
        symbol     = signal_dict.get("symbol", "")

        prompt = (
            f"Trading bot is entering a {direction} trade on {symbol}. "
            f"Model confidence: {confidence:.0%}. Market regime: {regime}. "
            "In ONE short sentence (max 15 words), explain why this trade makes sense."
        )
        explanation = self._ask(prompt, max_tokens=60)
        return explanation or f"{direction} signal at {confidence:.0%} confidence in {regime} regime."

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_decision(
        self, action: str, symbol: str, context: dict, result: dict
    ) -> None:
        """Append one JSONL record to the daily decisions log."""
        try:
            _DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
            today    = datetime.utcnow().strftime("%Y%m%d")
            log_file = _DECISIONS_DIR / f"ollama_decisions_{today}.jsonl"
            record   = {
                "ts":      datetime.utcnow().isoformat(),
                "action":  action,
                "symbol":  symbol,
                "context": context,
                "result":  result,
            }
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            _log.debug(f"_save_decision failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """
        Extract a JSON object from *text*, tolerating surrounding prose.
        Returns {} if no valid JSON object is found.
        """
        if not text:
            return {}
        # Try raw parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Locate first '{' … last '}' block
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}
