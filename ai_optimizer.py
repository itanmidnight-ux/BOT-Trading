"""
Integracion opcional con modelos gratuitos de OpenRouter para sugerir ajustes
de parametros a partir del historial reciente de trades.

Diseño deliberado: este modulo es puramente ASESOR. Nunca escribe directamente
sobre `config` ni sobre ordenes en curso — solo produce sugerencias acotadas
(clampeadas a +/- AI_OPTIMIZER_MAX_PARAM_SHIFT_PCT del valor actual) y las deja
en `logs/ai_suggested_params.json` para revision humana antes de aplicarlas.
Aplicar automaticamente sugerencias de un LLM sobre parametros de riesgo (SL,
trailing, tamaño de posicion) sin supervision seria un riesgo de estabilidad
inaceptable para un sistema que mueve dinero real; por eso el "auto-mejora" se
detiene en la sugerencia, no en la ejecucion.

Si AI_OPTIMIZER_ENABLED=False (default) o no hay OPENROUTER_API_KEY, este
modulo no hace ninguna llamada de red y el resto del bot funciona igual.
"""
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)

# Parametros que el modelo puede sugerir ajustar. Deliberadamente NO incluye
# limites duros de seguridad (MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT, MAX_OPEN_POSITIONS).
TUNABLE_PARAMS = {
    "CONFLUENCE_MIN_AVG_CONFIDENCE": config.CONFLUENCE_MIN_AVG_CONFIDENCE,
    "ATR_SL_MULTIPLIER": config.ATR_SL_MULTIPLIER,
    "ATR_TP1_MULTIPLIER": config.ATR_TP1_MULTIPLIER,
    "ATR_TP2_MULTIPLIER": config.ATR_TP2_MULTIPLIER,
    "ATR_TRAIL_DISTANCE_MULTIPLIER": config.ATR_TRAIL_DISTANCE_MULTIPLIER,
    "PROFIT_GIVEBACK_TOLERANCE_PCT": config.PROFIT_GIVEBACK_TOLERANCE_PCT,
}


@dataclass
class TradeStatsSummary:
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_rr: float
    max_drawdown_pct: float


def _clamp(param: str, suggested: float) -> float:
    current = TUNABLE_PARAMS[param]
    max_shift = abs(current) * (config.AI_OPTIMIZER_MAX_PARAM_SHIFT_PCT / 100.0)
    return max(current - max_shift, min(current + max_shift, suggested))


def _build_prompt(stats: TradeStatsSummary) -> str:
    return (
        "Sos un asistente de ajuste de parametros para un bot de trading algoritmico "
        "en XAUUSD 1m. Con base en estas metricas de los ultimos trades:\n"
        f"{json.dumps(asdict(stats), ensure_ascii=False)}\n\n"
        f"Parametros actuales (ajustables): {json.dumps(TUNABLE_PARAMS, ensure_ascii=False)}\n\n"
        "Responde EXCLUSIVAMENTE con un objeto JSON plano {\"PARAM\": nuevo_valor, ...} "
        "sugiriendo nuevos valores para cero o mas de esos parametros (mismo tipo numerico). "
        "No incluyas texto fuera del JSON. Si no hay ajuste claro que sugerir, responde {}."
    )


def request_suggestions(stats: TradeStatsSummary) -> Dict[str, float]:
    """Devuelve un dict de sugerencias YA CLAMPEADAS. Vacio si esta deshabilitado,
    falta la API key, o la llamada/parseo falla por cualquier motivo (nunca
    lanza excepcion hacia el caller: un fallo aca no debe tumbar el bot)."""
    if not config.AI_OPTIMIZER_ENABLED:
        return {}
    if not config.OPENROUTER_API_KEY:
        logger.warning("AI_OPTIMIZER_ENABLED=True pero falta OPENROUTER_API_KEY; se omite.")
        return {}

    try:
        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": _build_prompt(stats)}],
                "temperature": 0.2,
            },
            timeout=15,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        raw_suggestions = json.loads(content)
    except Exception as e:  # noqa: BLE001 - cualquier fallo de red/parseo es no-fatal aqui
        logger.warning("ai_optimizer: fallo al obtener sugerencias (%s); se ignora este ciclo.", e)
        return {}

    clamped = {}
    for param, value in raw_suggestions.items():
        if param not in TUNABLE_PARAMS:
            continue
        try:
            clamped[param] = _clamp(param, float(value))
        except (TypeError, ValueError):
            continue

    if clamped:
        _persist_suggestions(clamped)
    return clamped


def _persist_suggestions(suggestions: Dict[str, float]) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / "ai_suggested_params.json"
    path.write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
    logger.info("Sugerencias de ai_optimizer guardadas en %s (revision manual requerida): %s", path, suggestions)
