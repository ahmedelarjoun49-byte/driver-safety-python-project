from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.models import FramePacket

Point = tuple[float, float]


@dataclass(slots=True)
class FaceObservation:
    bbox: tuple[int, int, int, int]
    landmarks: dict[str, list[Point]]
    confidence: float = 1.0
    provider: str = "unknown"
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def all_points(self) -> list[Point]:
        points: list[Point] = []
        for group in self.landmarks.values():
            points.extend(group)
        return points


class FaceLandmarkDetector(Protocol):
    provider: str

    def detect(self, packet: FramePacket) -> list[FaceObservation]: ...


class ModelNotAvailableError(RuntimeError):
    pass


class HaarFaceDetector:
    provider = "haar"

    def __init__(self, cascade_path: Path | None = None) -> None:
        # Base paths layout resolution
        base_dir = Path(__file__).parents[1] if "__file__" in locals() else Path(".")
        legacy_dir = None
        
        possible_dirs = [
            Path("ai-driver-safety/legacy/haar-cascade-files"),
            Path("legacy/haar-cascade-files"),
            base_dir / "legacy/haar-cascade-files"
        ]
        
        for d in possible_dirs:
            if d.resolve().absolute().exists():
                legacy_dir = d.resolve().absolute()
                break
                
        if legacy_dir is None:
            legacy_dir = possible_dirs[0].resolve().absolute()

        # Load main face classifier
        self.face_classifier = cv2.CascadeClassifier(str(legacy_dir / "haarcascade_frontalface_default.xml"))
        
        # Load secondary feature classifiers for dynamic tracking states
        self.left_eye_classifier = cv2.CascadeClassifier(str(legacy_dir / "haarcascade_lefteye_2splits.xml"))
        self.right_eye_classifier = cv2.CascadeClassifier(str(legacy_dir / "haarcascade_righteye_2splits.xml"))
        self.smile_classifier = cv2.CascadeClassifier(str(legacy_dir / "haarcascade_smile.xml"))

        if self.face_classifier.empty():
            raise ModelNotAvailableError(f"Cannot load core Haar cascade files from layout location: {legacy_dir}")

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        frame = packet.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
        
        observations: list[FaceObservation] = []
        for x, y, w, h in faces[:1]:
            # ROI (Region of Interest) targeting top/bottom segments of detected facial bounds
            face_gray = gray[y:y+h, x:x+w]
            
            # Sub-divide regions for specialized eye tracking zones to eliminate false positives
            eye_region_h = int(h * 0.55)
            left_eye_zone = face_gray[0:eye_region_h, 0:int(w * 0.5)]
            right_eye_zone = face_gray[0:eye_region_h, int(w * 0.5):w]
            mouth_zone = face_gray[int(h * 0.55):h, 0:w]
            
            # Detect true open states dynamically based on raw cascade matching hits
            left_eyes = self.left_eye_classifier.detectMultiScale(left_eye_zone, scaleFactor=1.1, minNeighbors=3)
            right_eyes = self.right_eye_classifier.detectMultiScale(right_eye_zone, scaleFactor=1.1, minNeighbors=3)
            smiles = self.smile_classifier.detectMultiScale(mouth_zone, scaleFactor=1.15, minNeighbors=6)
            
            # Determine dynamic state tracking flags 
            left_eye_open = len(left_eyes) > 0
            right_eye_open = len(right_eyes) > 0
            mouth_yawning = len(smiles) > 0

            cx = int(x + w / 2)
            cy = int(y + h / 2)
            
            observations.append(
                FaceObservation(
                    bbox=(int(x), int(y), int(w), int(h)),
                    landmarks=_approximate_face_landmarks(
                        cx, cy, 
                        eye_open=(left_eye_open or right_eye_open), 
                        mouth_open=mouth_yawning, 
                        scale=w / 150
                    ),
                    provider=self.provider,
                    confidence=0.65,
                )
            )
        return observations


class MediaPipeFaceLandmarker:
    provider = "mediapipe"

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(f"MediaPipe face landmarker model missing: {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            raise ModelNotAvailableError("MediaPipe is not installed.") from exc

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(image, int(packet.timestamp * 1000))
        if not result.face_landmarks:
            return []
        h, w = packet.frame.shape[:2]
        landmarks = result.face_landmarks[0]
        points = [(landmark.x * w, landmark.y * h) for landmark in landmarks]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bbox = (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(w, max(xs)) - max(0, min(xs))),
            int(min(h, max(ys)) - max(0, min(ys))),
        )
        groups = {
            "left_eye": _pick(points, [33, 160, 158, 133, 153, 144]),
            "right_eye": _pick(points, [362, 385, 387, 263, 373, 380]),
            "mouth": _pick(points, [61, 81, 13, 291, 14, 178]),
            "pose": _pick(points, [1, 33, 263, 61, 291]),
        }
        return [FaceObservation(bbox=bbox, landmarks=groups, provider=self.provider)]


def create_face_detector(config: DriverSafetyConfig) -> FaceLandmarkDetector:
    provider = config.vision.provider.lower()
    if provider in {"mediapipe", "auto"}:
        try:
            return MediaPipeFaceLandmarker(Path(config.vision.face_landmarker_model))
        except ModelNotAvailableError:
            if provider == "mediapipe" and not config.vision.fallback_to_haar:
                raise
    if provider in {"haar", "opencv", "auto", "mediapipe"} and config.vision.fallback_to_haar:
        return HaarFaceDetector()
    raise ModelNotAvailableError(f"Unsupported face detector provider: {config.vision.provider}")


def _pick(points: list[Point], indexes: list[int]) -> list[Point]:
    return [points[index] for index in indexes if index < len(points)]


def _approximate_face_landmarks(
    cx: int,
    cy: int,
    *,
    eye_open: bool,
    mouth_open: bool,
    scale: float = 1.0,
) -> dict[str, list[Point]]:
    eye_half_w = 22 * scale
    eye_v = (8 if eye_open else 1.5) * scale
    mouth_half_w = 34 * scale
    mouth_v = (26 if mouth_open else 5) * scale
    
    left_eye_cx = cx - int(38 * scale)
    right_eye_cx = cx + int(38 * scale)
    eye_y = cy - int(32 * scale)
    mouth_y = cy + int(48 * scale)
    
    return {
        "left_eye": _eye_points(left_eye_cx, eye_y, eye_half_w, eye_v),
        "right_eye": _eye_points(right_eye_cx, eye_y, eye_half_w, eye_v),
        "mouth": [
            (cx - mouth_half_w, mouth_y),
            (cx - mouth_half_w / 2, mouth_y - mouth_v),
            (cx + mouth_half_w / 2, mouth_y - mouth_v),
            (cx + mouth_half_w, mouth_y),
            (cx + mouth_half_w / 2, mouth_y + mouth_v),
            (cx - mouth_half_w / 2, mouth_y + mouth_v),
        ],
        "pose": [(cx, cy), (left_eye_cx, eye_y), (right_eye_cx, eye_y)],
    }


def _eye_points(cx: float, cy: float, half_w: float, vertical: float) -> list[Point]:
    return [
        (cx - half_w, cy),
        (cx - half_w / 2, cy - vertical),
        (cx + half_w / 2, cy - vertical),
        (cx + half_w, cy),
        (cx + half_w / 2, cy + vertical),
        (cx - half_w / 2, cy + vertical),
    ]