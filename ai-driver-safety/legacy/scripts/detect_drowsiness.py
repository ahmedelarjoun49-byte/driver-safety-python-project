from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Optional

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

from driver_safety.core.models import DetectionEvent, DriverState


@dataclass(slots=True)
class EyeConfig:

    predictor_path: str = (
        "shape_predictor_68_face_landmarks.dat"
    )

    ear_threshold: float = 0.25

    consecutive_frames: int = 48



class EyeDrowsinessDetector:

    """
    Detects eye closure and drowsiness
    using Eye Aspect Ratio (EAR).
    """


    def __init__(
        self,
        config: EyeConfig | None = None,
    ) -> None:

        self.config = config or EyeConfig()

        predictor_file = Path(
            self.config.predictor_path
        )

        if not predictor_file.exists():
            raise FileNotFoundError(
                f"Missing landmark model: {predictor_file}"
            )


        self.detector = (
            dlib.get_frontal_face_detector()
        )

        self.predictor = (
            dlib.shape_predictor(
                str(predictor_file)
            )
        )


        self.left_start, self.left_end = (
            face_utils
            .FACIAL_LANDMARKS_IDXS["left_eye"]
        )

        self.right_start, self.right_end = (
            face_utils
            .FACIAL_LANDMARKS_IDXS["right_eye"]
        )


        self.closed_counter = 0



    def process(
        self,
        frame: np.ndarray,
    ) -> dict[str, float]:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )


        faces = self.detector(
            gray,
            0,
        )


        if not faces:

            self.closed_counter = 0

            return {
                "eyes_closed": 0.0,
                "drowsy": 0.0,
            }



        face = faces[0]


        landmarks = self.predictor(
            gray,
            face,
        )


        points = face_utils.shape_to_np(
            landmarks
        )


        left_eye = points[
            self.left_start:self.left_end
        ]

        right_eye = points[
            self.right_start:self.right_end
        ]


        ear = (
            self._eye_aspect_ratio(left_eye)
            +
            self._eye_aspect_ratio(right_eye)
        ) / 2



        eyes_closed = 0.0


        if ear < self.config.ear_threshold:

            self.closed_counter += 1

            eyes_closed = min(
                self.closed_counter /
                self.config.consecutive_frames,
                1.0,
            )

        else:

            self.closed_counter = 0



        return {

            "eyes_closed": round(
                eyes_closed,
                3,
            ),

            "drowsy": round(
                self._calculate_drowsiness(
                    eyes_closed,
                    ear,
                ),
                3,
            ),
        }



    def _calculate_drowsiness(
        self,
        closed_score: float,
        ear: float,
    ) -> float:

        """
        Converts EAR behavior into a drowsiness score.
        """

        ear_factor = max(
            0,
            (
                self.config.ear_threshold - ear
            )
            /
            self.config.ear_threshold,
        )


        return min(
            1.0,
            (
                closed_score * 0.7
                +
                ear_factor * 0.3
            ),
        )



    @staticmethod
    def _eye_aspect_ratio(
        eye: np.ndarray,
    ) -> float:

        vertical_1 = dist.euclidean(
            eye[1],
            eye[5],
        )

        vertical_2 = dist.euclidean(
            eye[2],
            eye[4],
        )

        horizontal = dist.euclidean(
            eye[0],
            eye[3],
        )


        return (
            vertical_1 + vertical_2
        ) / (
            2.0 * horizontal
        )