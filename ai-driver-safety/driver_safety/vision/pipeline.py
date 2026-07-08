from __future__ import annotations

from time import perf_counter
import requests
import json

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.alerts import AlertPolicy
from driver_safety.core.models import (
    DetectionEvent,
    DriverState,
    FramePacket,
    ProcessedFrame,
    Severity,
)
from driver_safety.core.scoring import RiskScorer
from driver_safety.core.smoothing import SignalSmoother
from driver_safety.vision.landmarks import FaceLandmarkDetector, create_face_detector
from driver_safety.vision.metrics import (
    eye_aspect_ratio,
    horizontal_head_offset,
    mouth_aspect_ratio,
)
from driver_safety.vision.object_detector import (
    ObjectDetector,
    ObjectObservation,
    create_object_detector,
)


class DriverSafetyPipeline:
    def __init__(
        self,
        config: DriverSafetyConfig,
        *,
        face_detector: FaceLandmarkDetector | None = None,
        object_detector: ObjectDetector | None = None,
    ) -> None:
        self.config = config
        self.face_detector = face_detector or create_face_detector(config)
        self.object_detector = object_detector or create_object_detector(config)
        self.smoother = SignalSmoother(window_size=5)
        self.scorer = RiskScorer(config.signal_weights or None)
        self.alert_policy = AlertPolicy(config.runtime.alert_cooldown_seconds)
        self._closed_counter = 0
        self._yawn_counter = 0
        self._distracted_counter = 0
        self._missing_face_counter = 0
        self._phone_counter = 0
        self._phone_hold_counter = 0
        self._last_phone: ObjectObservation | None = None

        # Custom Cloud Rate Limiting Setup (Standalone Fallback)
        self._cloud_cooldowns = {}
        self._cooldown_duration = getattr(config.runtime, "alert_cooldown_seconds", 5)

        # Inline Credentials Configuration
        self.supabase_url = "https://ybcfbbclcamcccqfkgbi.supabase.co"
        self.supabase_key = "sb_publishable_gjLy3MxPn4NgPWPsmhk5kQ_E45BAsrU"

    def _send_cloud_alert(self, event_type: str, risk_score: float) -> None:
        """Self-contained method to stream alerts directly to your database without file dependencies."""
        url = f"{self.supabase_url}/rest/v1/incidents"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {
            "driver_name": "Ahmed",
            "event_type": str(event_type),
            "risk_index": float(risk_score)
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            if response.status_code == 201:
                print(f" [CLOUD LOG] Successfully uploaded alert: {event_type} ({risk_score})")
            else:
                print(f" [CLOUD ERROR] Remote database returned code: {response.status_code}")
        except Exception as e:
            print(f" [CLOUD FAILED] Network Connection Error: {e}")

    def process_frame(self, packet: FramePacket) -> ProcessedFrame:
        started = perf_counter()
        raw_signals = {
            "eyes_closed": 0.0,
            "drowsy": 0.0,
            "yawning": 0.0,
            "distracted": 0.0,
            "phone_use": 0.0,
        }
        events: list[DetectionEvent] = []
        face_bbox: tuple[int, int, int, int] | None = None
        landmarks: list[tuple[float, float]] = []
        face_observations = self.face_detector.detect(packet)

        if not face_observations:
            self._missing_face_counter += 1
            if self._missing_face_counter >= self.config.thresholds.missing_face_frames:
                raw_signals["distracted"] = 1.0
                events.append(
                    self._event(
                        packet,
                        "distracted",
                        DriverState.DISTRACTED,
                        1.0,
                        Severity.WARNING,
                        "Distracted: driver attention not trackable",
                    )
                )
        else:
            self._missing_face_counter = 0
            observation = face_observations[0]
            face_bbox = observation.bbox
            landmarks = observation.all_points
            raw_signals.update(self._face_signals(packet, observation.bbox, observation.landmarks))
            events.extend(self._events_from_signals(packet, raw_signals, face_bbox, landmarks))

        object_observations = self.object_detector.detect(packet)
        phone_labels = {label.lower() for label in self.config.object_detector.phone_labels}
        best_phone = None
        display_objects: list[ObjectObservation] = []
        for obj in object_observations:
            if (
                obj.label.lower() in phone_labels
                and obj.confidence >= self.config.thresholds.phone_confidence
                and (best_phone is None or obj.confidence > best_phone.confidence)
            ):
                best_phone = obj

        if best_phone is None:
            if self._phone_hold_counter > 0 and self._last_phone is not None:
                self._phone_hold_counter -= 1
                best_phone = self._last_phone
            else:
                self._phone_counter = 0
                self._last_phone = None
        else:
            self._last_phone = best_phone
            self._phone_hold_counter = self.config.thresholds.phone_hold_frames

        if best_phone is not None:
            self._phone_counter += 1
            phone_gate = min(
                1.0, self._phone_counter / max(1, self.config.thresholds.phone_use_frames)
            )
            raw_signals["phone_use"] = best_phone.confidence * phone_gate
            if self._phone_counter >= self.config.thresholds.phone_use_frames:
                events.append(
                    self._event(
                        packet,
                        "phone_use",
                        DriverState.PHONE_USE,
                        raw_signals["phone_use"],
                        Severity.CRITICAL,
                        "Phone use detected while driving",
                        bbox=best_phone.bbox,
                        metadata={
                            "label": best_phone.label,
                            "provider": best_phone.provider,
                            "tracking": "raw" if best_phone in object_observations else "held",
                        },
                    )
                )
                display_objects.append(best_phone)

        signals = self.smoother.update(raw_signals)
        risk_score = self.scorer.score(signals)
        state = self.scorer.state_from_events(events, risk_score)
        latency_ms = (perf_counter() - started) * 1000
        return ProcessedFrame(
            packet=packet,
            state=state,
            risk_score=risk_score,
            signals=signals,
            events=events,
            latency_ms=latency_ms,
            face_bbox=face_bbox,
            landmarks=landmarks,
            objects=[obj.to_dict() for obj in display_objects],
        )

    def _face_signals(
        self,
        packet: FramePacket,
        bbox: tuple[int, int, int, int],
        landmarks: dict[str, list[tuple[float, float]]],
    ) -> dict[str, float]:
        thresholds = self.config.thresholds
        left_ear = eye_aspect_ratio(landmarks["left_eye"])
        right_ear = eye_aspect_ratio(landmarks["right_eye"])
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(landmarks["mouth"])
        head_offset = horizontal_head_offset(bbox, packet.frame.shape[1])

        self._closed_counter = self._closed_counter + 1 if ear < thresholds.eye_aspect_ratio else 0
        self._yawn_counter = self._yawn_counter + 1 if mar > thresholds.mouth_aspect_ratio else 0
        self._distracted_counter = (
            self._distracted_counter + 1 if head_offset > thresholds.head_offset else 0
        )

        return {
            "eyes_closed": min(1.0, self._closed_counter / max(1, thresholds.eye_closed_frames)),
            "drowsy": min(1.0, self._closed_counter / max(1, thresholds.drowsy_frames)),
            "yawning": min(1.0, self._yawn_counter / max(1, thresholds.yawn_frames)),
            "distracted": min(1.0, self._distracted_counter / max(1, thresholds.distracted_frames)),
            "phone_use": 0.0,
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "head_offset": round(head_offset, 4),
        }

    def _events_from_signals(
        self,
        packet: FramePacket,
        signals: dict[str, float],
        bbox: tuple[int, int, int, int],
        landmarks: list[tuple[float, float]],
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        thresholds = self.config.thresholds
        if self._closed_counter >= thresholds.eye_closed_frames:
            events.append(
                self._event(
                    packet,
                    "eyes_closed",
                    DriverState.EYES_CLOSED,
                    signals["eyes_closed"],
                    Severity.WARNING,
                    "Eyes closed beyond configured threshold",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._closed_counter >= thresholds.drowsy_frames:
            events.append(
                self._event(
                    packet,
                    "drowsy",
                    DriverState.DROWSY,
                    signals["drowsy"],
                    Severity.CRITICAL,
                    "Sustained eye closure indicates drowsiness",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._yawn_counter >= thresholds.yawn_frames:
            events.append(
                self._event(
                    packet,
                    "yawning",
                    DriverState.YAWNING,
                    signals["yawning"],
                    Severity.WARNING,
                    "Yawn detected from mouth landmarks",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._distracted_counter >= thresholds.distracted_frames:
            events.append(
                self._event(
                    packet,
                    "distracted",
                    DriverState.DISTRACTED,
                    signals["distracted"],
                    Severity.WARNING,
                    "Head pose indicates driver is looking away",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        return events

    def _event(
        self,
        packet: FramePacket,
        signal: str,
        state: DriverState,
        score: float,
        severity: Severity,
        message: str,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
        metadata: dict[str, float | str] | None = None,
    ) -> DetectionEvent:
        event = DetectionEvent(
            timestamp=packet.timestamp,
            frame_index=packet.frame_index,
            signal=signal,
            state=state,
            score=round(float(score), 4),
            severity=severity,
            message=message,
            bbox=bbox,
            landmarks=landmarks or [],
            metadata=metadata or {},
        )

        # Standalone Alert Cooldown Validation
        if state in [DriverState.PHONE_USE, DriverState.DROWSY, DriverState.DISTRACTED]:
            last_triggered = self._cloud_cooldowns.get(state.name, 0.0)
            if packet.timestamp - last_triggered >= self._cooldown_duration:
                self._cloud_cooldowns[state.name] = packet.timestamp
                clean_event_name = "PHONE_USE" if state == DriverState.PHONE_USE else state.name
                self._send_cloud_alert(event_type=clean_event_name, risk_score=score)

        return event


def create_pipeline(config: DriverSafetyConfig | None = None) -> DriverSafetyPipeline:
    """Factory helper function to safely generate an operational pipeline."""
    if config is None:
        raise ValueError("A valid DriverSafetyConfig configuration instance must be supplied.")
    return DriverSafetyPipeline(config)