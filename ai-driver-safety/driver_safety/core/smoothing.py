from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class RollingWindow:
    """
    Maintains a fixed-size rolling buffer with O(1) mean calculation.
    """

    size: int
    values: deque[float] = field(default_factory=deque)
    _current_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Rolling window size must be greater than zero")

    def add(self, value: float) -> float:
        """
        Adds a new value and returns the updated rolling average.
        """

        numeric_value = float(value)

        self.values.append(numeric_value)
        self._current_sum += numeric_value

        if len(self.values) > self.size:
            removed_value = self.values.popleft()
            self._current_sum -= removed_value

        return self.mean

    @property
    def mean(self) -> float:
        """
        Returns current rolling average.
        """

        if not self.values:
            return 0.0

        return self._current_sum / len(self.values)

    def reset(self) -> None:
        """
        Clears the window state.
        """

        self.values.clear()
        self._current_sum = 0.0


class SignalSmoother:
    """
    Applies temporal smoothing to detection signals.

    Useful for reducing frame-to-frame fluctuations
    from computer vision models.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero"
            )

        self.window_size = window_size

        self._windows: dict[str, RollingWindow] = defaultdict(
            lambda: RollingWindow(size=self.window_size)
        )

    def update(
        self,
        signals: dict[str, float],
    ) -> dict[str, float]:
        """
        Updates signal history and returns smoothed values.
        """

        return {
            name: self._windows[name].add(value)
            for name, value in signals.items()
        }

    def reset(self) -> None:
        """
        Resets all tracked signals.
        """

        for window in self._windows.values():
            window.reset()

        self._windows.clear()