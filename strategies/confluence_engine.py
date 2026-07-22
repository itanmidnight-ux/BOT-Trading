"""
Motor de confluencia: solo autoriza un trade cuando suficientes estrategias
habilitadas coinciden en direccion, con confianza promedio por encima del umbral.
Esto es lo que evita que el bot opere con cada señal aislada y reduce ruido/
overtrading de baja calidad, sin sacrificar la frecuencia (varias estrategias
rapidas de 1m siguen generando muchas oportunidades reales por dia).
"""
from typing import Dict, List

import pandas as pd

import config
from strategies import (
    ema_trend_cross,
    rsi_bollinger_reversion,
    fractal_breakout,
    vwap_momentum,
)
from strategies.base import Direction, Signal

_STRATEGY_MODULES = {
    "ema_trend_cross": ema_trend_cross,
    "rsi_bollinger_reversion": rsi_bollinger_reversion,
    "fractal_breakout": fractal_breakout,
    "vwap_momentum": vwap_momentum,
}


class ConfluenceResult:
    def __init__(self, direction: Direction, confidence: float,
                 agreeing: List[Signal], all_signals: List[Signal]):
        self.direction = direction
        self.confidence = confidence
        self.agreeing = agreeing
        self.all_signals = all_signals

    @property
    def is_actionable(self) -> bool:
        return self.direction != Direction.FLAT

    def as_dict(self) -> Dict:
        return {
            "direction": int(self.direction),
            "confidence": round(self.confidence, 4),
            "agreeing_strategies": [s.strategy_name for s in self.agreeing],
            "all_signals": [
                {"name": s.strategy_name, "direction": int(s.direction), "confidence": round(s.confidence, 4)}
                for s in self.all_signals
            ],
        }


def run_all_strategies(df: pd.DataFrame) -> List[Signal]:
    signals = []
    for name, enabled in config.STRATEGIES_ENABLED.items():
        if not enabled:
            continue
        module = _STRATEGY_MODULES[name]
        signals.append(module.generate_signal(df))
    return signals


def evaluate_confluence(df: pd.DataFrame) -> ConfluenceResult:
    signals = run_all_strategies(df)

    buys = [s for s in signals if s.direction == Direction.BUY]
    sells = [s for s in signals if s.direction == Direction.SELL]

    def build(direction: Direction, group: List[Signal]) -> ConfluenceResult:
        if len(group) < config.CONFLUENCE_MIN_STRATEGIES:
            return ConfluenceResult(Direction.FLAT, 0.0, [], signals)
        avg_conf = sum(s.confidence for s in group) / len(group)
        if avg_conf < config.CONFLUENCE_MIN_AVG_CONFIDENCE:
            return ConfluenceResult(Direction.FLAT, 0.0, [], signals)
        return ConfluenceResult(direction, avg_conf, group, signals)

    buy_result = build(Direction.BUY, buys)
    sell_result = build(Direction.SELL, sells)

    if buy_result.is_actionable and sell_result.is_actionable:
        # Señales contradictorias con quorum en ambos lados: no operar (ambiguedad real del mercado).
        return ConfluenceResult(Direction.FLAT, 0.0, [], signals)
    if buy_result.is_actionable:
        return buy_result
    if sell_result.is_actionable:
        return sell_result
    return ConfluenceResult(Direction.FLAT, 0.0, [], signals)
