from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from driver_safety.core.models import DetectionEvent


@dataclass(slots=True)
class Alert:
    timestamp: float
    signal: str
    message: str
    severity: str


class AlertPolicy:
    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        history_window: float = 20.0,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.history_window = history_window

        self._last_alert_by_signal: dict[str, float] = {}
        self._history: dict[str, deque[float]] = defaultdict(deque)

    def _update_history(self, signal: str, timestamp: float) -> int:
        history = self._history[signal]
        history.append(timestamp)

        while history and timestamp - history[0] > self.history_window:
            history.popleft()

        return len(history)

    def evaluate(self, events: list[DetectionEvent]) -> list[Alert]:
        alerts: list[Alert] = []

        signals_this_frame = {event.signal for event in events}

        for event in events:
            last_alert = self._last_alert_by_signal.get(event.signal)

            if (
                last_alert is not None
                and event.timestamp - last_alert < self.cooldown_seconds
            ):
                continue

            occurrences = self._update_history(event.signal, event.timestamp)

            severity = event.severity.value
            message = event.message

            # Escalate repeated behaviour
            if occurrences >= 5:
                severity = "high"
                message += " (Repeated behaviour detected)"

            # Combined dangerous behaviours
            if (
                "phone_use" in signals_this_frame
                and "drowsy" in signals_this_frame
            ):
                severity = "critical"
                message = (
                    "Phone use detected while driver appears drowsy."
                )

            if (
                "eyes_closed" in signals_this_frame
                and "yawning" in signals_this_frame
            ):
                severity = "critical"
                message = (
                    "Fatigue indicators detected simultaneously."
                )

            self._last_alert_by_signal[event.signal] = event.timestamp

            alerts.append(
                Alert(
                    timestamp=event.timestamp,
                    signal=event.signal,
                    message=message,
                    severity=severity,
                )
            )

        return alerts