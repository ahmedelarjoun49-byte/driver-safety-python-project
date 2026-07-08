from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from driver_safety.core.models import DriverState, ProcessedFrame

Array = NDArray[Any]

# Modern, vibrant color palettes (BGR format)
STATE_COLORS = {
    DriverState.ATTENTIVE: (124, 197, 62),   # Crisp Emerald Green
    DriverState.EYES_CLOSED: (36, 174, 222), # Safety Orange
    DriverState.DROWSY: (38, 64, 230),      # High-Visibility Crimson Red
    DriverState.YAWNING: (235, 167, 46),     # Warning Amber
    DriverState.DISTRACTED: (42, 42, 238),   # Alert Coral Red
    DriverState.PHONE_USE: (42, 42, 238),    # Alert Coral Red
}

LOGO_PATHS = ()


class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.path}")

    def write(self, frame: Array) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def draw_overlay(processed: ProcessedFrame) -> Array:
    frame = processed.packet.frame.copy()
    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))
    
    # 1. Left Telemetry HUD Panel (Modernized Layout)
    _draw_panel(frame, 15, 15, 330, 140, alpha=0.75)
    _draw_brand(frame, 30, 42)
    
    # Dynamic Status Label Configuration
    cv2.putText(
        frame,
        processed.state.value.replace("_", " ").upper(),
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )
    
    # Risk and Latency Tracking Display
    cv2.putText(
        frame,
        f"RISK INDEX: {processed.risk_score:.2f}  |  {processed.latency_ms:.1f} ms",
        (30, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 225, 225),
        1,
        cv2.LINE_AA,
    )
    
    # 2. Main Target Tracking Overlays
    if processed.face_bbox:
        x, y, bw, bh = processed.face_bbox
        # Draw a sleeker, thinner, high-precision bounding box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 2, cv2.LINE_AA)
        
        # Subtle corners to make the box frame look futuristic
        corner_len = int(min(bw, bh) * 0.15)
        cv2.line(frame, (x, y), (x + corner_len, y), state_color, 4)
        cv2.line(frame, (x, y), (x, y + corner_len), state_color, 4)
        cv2.line(frame, (x + bw, y), (x + bw - corner_len, y), state_color, 4)
        cv2.line(frame, (x + bw, y), (x + bw, y + corner_len), state_color, 4)

    # Clean geometric representation of face points
    for point in processed.landmarks[:: max(1, len(processed.landmarks) // 36 or 1)]:
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 230, 110), -1)
        
    # Tracked Objects Overlay (e.g., cell phones)
    for obj in processed.objects:
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (65, 65, 235), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(obj["label"]).upper(),
            (x + 4, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (65, 65, 235),
            2,
            cv2.LINE_AA,
        )
        
    # 3. Sidebar Right Signal Telemetry Bars
    _draw_signal_bars(frame, processed, w, h)
    
    # 4. System Live Log Notifications Bottom Left
    for idx, event in enumerate(processed.events[:3]):
        y_pos = h - 95 + idx * 24
        # Add tiny indicator circle for logs
        cv2.circle(frame, (25, y_pos - 4), 3, state_color, -1)
        cv2.putText(
            frame,
            event.message,
            (36, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        
    return cast(Array, frame)


def _draw_panel(frame: Array, x: int, y: int, w: int, h: int, *, alpha: float) -> None:
    """Draws a premium dark translucent overlay panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (26, 21, 18), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (55, 50, 48), 1, cv2.LINE_AA) # Clean subtle border
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_brand(frame: Array, x: int, y: int) -> None:
    """Renders the stylized application branding logo text."""
    cv2.putText(
        frame,
        "DRIVESAFE-AI",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (250, 252, 252),
        2,
        cv2.LINE_AA,
    )


@lru_cache(maxsize=1)
def _load_logo() -> Array | None:
    return None


def _blend_rgba(frame: Array, overlay: Array, x: int, y: int) -> None:
    pass


def _draw_signal_bars(frame: Array, processed: ProcessedFrame, width: int, height: int) -> None:
    """Renders a sleek HUD bar monitor array on the right side of the video frame."""
    labels = ["eyes_closed", "drowsy", "yawning", "distracted", "phone_use"]
    start_x = max(20, width - 245)
    start_y = 30
    
    _draw_panel(frame, start_x - 15, start_y - 15, 230, 165, alpha=0.72)
    
    for idx, label in enumerate(labels):
        value = min(1.0, max(0.0, processed.signals.get(label, 0.0)))
        y = start_y + idx * 26
        
        # Text Label
        cv2.putText(
            frame,
            label.replace("_", " ").upper(),
            (start_x, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (215, 220, 220),
            1,
            cv2.LINE_AA,
        )
        
        # Background Bar
        bar_x = start_x + 110
        bar_w = 95
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 10), (55, 50, 48), -1)
        
        # Dynamic Foreground Active Fill Bar
        fill_w = int(bar_w * value)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bar_x, y),
                (bar_x + fill_w, y + 10),
                _signal_color(label, value),
                -1,
            )


def _signal_color(label: str, value: float) -> tuple[int, int, int]:
    """Computes dynamic indicator bar colors based on risk severity thresholds."""
    if value < 0.45:
        return (124, 197, 62) # Clear Green
    if label in ("distracted", "phone_use", "drowsy") and value > 0.70:
        return (42, 42, 238)  # Critical Red Alert
    return (46, 167, 235)     # Warning Amber/Orange