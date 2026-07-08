from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class RollingWindow:
    size: int
    values: deque[float] = field(default_factory=deque)
    _current_sum: float = 0.0  # Tracks sum dynamically in O(1) time

    def add(self, value: float) -> float:
        val = float(value)
        self.values.append(val)
        self._current_sum += val
        
        # Maintain window size constraint without loop overhead
        if len(self.values) > self.size:
            self._current_sum -= self.values.popleft()
            
        return self.mean

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self._current_sum / len(self.values)


class SignalSmoother:
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        # Optimized instantiation using explicit factories
        self._windows: dict[str, RollingWindow] = defaultdict(
            lambda: RollingWindow(size=window_size)
        )

    def update(self, signals: dict[str, float]) -> dict[str, float]:
        """Smooths incoming metrics to eliminate frame jitter and tracking noise."""
        return {
            name: self._windows[name].add(value) 
            for name, value in signals.items()
        }