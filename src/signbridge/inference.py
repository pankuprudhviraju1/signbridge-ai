"""Framework-neutral inference utilities used around a neural classifier."""

from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

Point = tuple[float, float, float]


def normalize_landmarks(points: Sequence[Point]) -> list[Point]:
    """Center landmarks on the wrist and scale by the furthest joint."""
    if len(points) != 21:
        raise ValueError("expected 21 hand landmarks")
    origin = points[0]
    centered = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in points]
    scale = max((sqrt(x * x + y * y + z * z) for x, y, z in centered), default=0)
    if scale <= 1e-9:
        raise ValueError("landmarks have zero scale")
    return [(x / scale, y / scale, z / scale) for x, y, z in centered]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    stable: bool


class TemporalRecognizer:
    """Reject uncertain predictions and emit only temporally stable labels."""

    def __init__(self, window: int = 5, confidence: float = 0.72, consensus: float = 0.6):
        if window < 1 or not 0 <= confidence <= 1 or not 0 < consensus <= 1:
            raise ValueError("invalid recognizer thresholds")
        self.window, self.confidence, self.consensus = window, confidence, consensus
        self._labels: deque[str] = deque(maxlen=window)

    def update(self, probabilities: Mapping[str, float]) -> Prediction:
        if not probabilities:
            raise ValueError("probabilities cannot be empty")
        label, score = max(probabilities.items(), key=lambda item: item[1])
        if not 0 <= score <= 1:
            raise ValueError("probability must be between 0 and 1")
        accepted = label if score >= self.confidence else "UNKNOWN"
        self._labels.append(accepted)
        winner, count = Counter(self._labels).most_common(1)[0]
        stable = len(self._labels) == self.window and count / self.window >= self.consensus
        return Prediction(winner if stable else accepted, score, stable)

    def reset(self) -> None:
        self._labels.clear()
