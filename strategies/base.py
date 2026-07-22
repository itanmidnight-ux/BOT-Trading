"""Tipos comunes usados por todas las estrategias."""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict


class Direction(IntEnum):
    SELL = -1
    FLAT = 0
    BUY = 1


@dataclass
class Signal:
    """Salida estandar de cada estrategia individual."""
    strategy_name: str
    direction: Direction
    confidence: float          # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.direction != Direction.FLAT and self.confidence > 0


def flat_signal(strategy_name: str, reason: str = "") -> Signal:
    return Signal(strategy_name=strategy_name, direction=Direction.FLAT, confidence=0.0,
                  details={"reason": reason} if reason else {})
