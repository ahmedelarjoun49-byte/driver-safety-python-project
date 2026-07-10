from __future__ import annotations

from collections import Counter
from statistics import mean

from driver_safety.core.models import DetectionEvent, DriverState, SessionSummary


DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "eyes_closed": 0.34,
    "drowsy": 0.54,
    "yawning": 0.22,
    "distracted": 0.34,
    "phone_use": 0.64,
    "sensor_drowsiness": 0.58,
    "lane_drift": 0.32,
    "short_time_to_collision": 0.48,
    "hard_maneuver": 0.26,
    "speeding": 0.20,
}

FUSION_MODEL_NAME = "driver-risk-fusion-v1"

UNSAFE_THRESHOLD = 0.45
HIGH_RISK_THRESHOLD = 0.55
CONTIGUOUS_GAP_SECONDS = 1.25


class RiskScorer:
    """
    Combines multiple driver safety signals into a unified risk score.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_SIGNAL_WEIGHTS)

    def score(self, signals: dict[str, float]) -> float:
        """
        Calculates overall driver risk using weighted noisy-or fusion.
        """

        evidence = [
            _clamp(signals.get(signal, 0.0)) * weight
            for signal, weight in self.weights.items()
            if signals.get(signal, 0.0) > 0
        ]

        risk = _noisy_or(evidence)
        risk += self._cross_signal_boost(signals)

        return round(_clamp(risk), 4)

    def fusion_channels(self, signals: dict[str, float]) -> dict[str, float]:
        """
        Groups individual signals into higher-level risk categories.
        """

        return {
            "vision_fatigue": max(
                _clamped(signals, "drowsy"),
                _clamped(signals, "eyes_closed") * 0.85,
                _clamped(signals, "yawning") * 0.55,
            ),
            "visual_distraction": max(
                _clamped(signals, "distracted"),
                _clamped(signals, "phone_use"),
            ),
            "physiology_fatigue": _clamped(signals, "sensor_drowsiness"),
            "vehicle_risk": max(
                _clamped(signals, "short_time_to_collision"),
                _clamped(signals, "lane_drift") * 0.8,
                _clamped(signals, "hard_maneuver") * 0.7,
                _clamped(signals, "speeding") * 0.55,
            ),
        }

    def _cross_signal_boost(self, signals: dict[str, float]) -> float:
        """
        Adds extra risk when multiple dangerous signals happen together.
        """

        channels = self.fusion_channels(signals)

        drowsy = _clamped(signals, "drowsy")
        eyes_closed = _clamped(signals, "eyes_closed")
        yawning = _clamped(signals, "yawning")

        boost = 0.0

        if drowsy >= 0.75 and eyes_closed >= 0.75:
            boost += 0.08

        if drowsy >= 0.6 and yawning >= 0.5:
            boost += 0.08

        if (
            channels["vision_fatigue"] >= 0.6
            and channels["physiology_fatigue"] >= 0.6
        ):
            boost += 0.14

        if (
            channels["vision_fatigue"] >= 0.6
            and channels["vehicle_risk"] >= 0.5
        ):
            boost += 0.10

        if (
            channels["visual_distraction"] >= 0.6
            and channels["vehicle_risk"] >= 0.5
        ):
            boost += 0.14

        return boost

    def state_from_events(
        self,
        events: list[DetectionEvent],
        risk_score: float,
    ) -> DriverState:
        """
        Determines the dominant driver state from detected events.
        """

        priority = (
            DriverState.PHONE_USE,
            DriverState.DROWSY,
            DriverState.EYES_CLOSED,
            DriverState.YAWNING,
            DriverState.DISTRACTED,
        )

        active_states = {event.state for event in events}

        for state in priority:
            if state in active_states:
                return state

        return (
            DriverState.DISTRACTED
            if risk_score >= HIGH_RISK_THRESHOLD
            else DriverState.ATTENTIVE
        )

    def summarize(
        self,
        *,
        session_id: str,
        source: str,
        duration_seconds: float,
        processed_frames: int,
        events: list[DetectionEvent],
        frame_scores: list[tuple[float, float]],
        metrics: dict[str, float | int | str],
    ) -> SessionSummary:

        event_counts = Counter(event.signal for event in events)

        risk_timeline = [
            {
                "timestamp": round(timestamp, 3),
                "risk_score": round(score, 4),
            }
            for timestamp, score in frame_scores
        ]

        unsafe_times = [
            timestamp
            for timestamp, score in frame_scores
            if score >= UNSAFE_THRESHOLD
        ]

        signal_scores: dict[str, list[float]] = {}

        for event in events:
            signal_scores.setdefault(event.signal, []).append(event.score)

        confidence_distribution = {
            signal: round(mean(scores), 4)
            for signal, scores in sorted(signal_scores.items())
            if scores
        }

        summary_metrics = dict(metrics)
        summary_metrics.setdefault(
            "fusion_model",
            FUSION_MODEL_NAME,
        )

        return SessionSummary(
            session_id=session_id,
            source=source,
            duration_seconds=round(duration_seconds, 3),
            processed_frames=processed_frames,
            event_counts=dict(sorted(event_counts.items())),
            risk_timeline=risk_timeline,
            longest_unsafe_interval_seconds=round(
                _longest_contiguous_interval(unsafe_times),
                3,
            ),
            confidence_distribution=confidence_distribution,
            metrics=summary_metrics,
        )


def _clamped(signals: dict[str, float], name: str) -> float:
    """
    Safely retrieves and clamps a signal value.
    """

    return _clamp(signals.get(name, 0.0))


def _longest_contiguous_interval(timestamps: list[float]) -> float:
    """
    Calculates the longest continuous unsafe driving period.
    """

    if len(timestamps) < 2:
        return 0.0

    longest = 0.0
    start = previous = timestamps[0]

    for timestamp in timestamps[1:]:
        if timestamp - previous > CONTIGUOUS_GAP_SECONDS:
            longest = max(longest, previous - start)
            start = timestamp

        previous = timestamp

    return max(longest, previous - start)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _noisy_or(evidence: list[float]) -> float:
    """
    Probability fusion using noisy-or model.
    """

    probability = 1.0

    for value in evidence:
        probability *= 1.0 - _clamp(value)

    return 1.0 - probability