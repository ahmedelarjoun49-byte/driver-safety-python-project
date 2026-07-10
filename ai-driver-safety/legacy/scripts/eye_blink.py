from __future__ import annotations

from dataclasses import dataclass

import cv2
import dlib
import numpy as np

from imutils import face_utils
from scipy.spatial import distance as dist



@dataclass(slots=True)
class BlinkConfig:

    predictor_path: str = (
        "shape_predictor_68_face_landmarks.dat"
    )

    ear_threshold: float = 0.25

    closed_frames_required: int = 3

    blink_reset_frames: int = 6



class BlinkDetector:
    """
    Detects blinking and prolonged eye closure
    using Eye Aspect Ratio (EAR).
    """


    def __init__(
        self,
        config: BlinkConfig | None = None,
    ):

        self.config = config or BlinkConfig()


        self.detector = (
            dlib.get_frontal_face_detector()
        )


        self.predictor = (
            dlib.shape_predictor(
                self.config.predictor_path
            )
        )


        (
            self.left_start,
            self.left_end,
        ) = face_utils.FACIAL_LANDMARKS_IDXS[
            "left_eye"
        ]


        (
            self.right_start,
            self.right_end,
        ) = face_utils.FACIAL_LANDMARKS_IDXS[
            "right_eye"
        ]


        self.closed_counter = 0
        self.blink_count = 0
        self.open_counter = 0



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

            return {
                "blink": 0.0,
                "eyes_closed": 0.0,
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


        left_ear = self._ear(left_eye)
        right_ear = self._ear(right_eye)


        ear = (
            left_ear +
            right_ear
        ) / 2



        eyes_closed = 0.0
        blink = 0.0



        if ear < self.config.ear_threshold:

            self.closed_counter += 1
            self.open_counter = 0


            eyes_closed = min(
                self.closed_counter /
                self.config.closed_frames_required,
                1.0,
            )


        else:

            self.open_counter += 1


            if (
                self.closed_counter >=
                self.config.closed_frames_required
            ):

                self.blink_count += 1
                blink = 1.0


            if (
                self.open_counter >=
                self.config.blink_reset_frames
            ):

                self.closed_counter = 0



        return {

            "blink": blink,

            "eyes_closed": round(
                eyes_closed,
                3,
            ),

        }



    @staticmethod
    def _ear(
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


        if horizontal == 0:
            return 0.0


        return (
            vertical_1 +
            vertical_2
        ) / (
            2.0 *
            horizontal
        )



    def reset(self) -> None:

        self.closed_counter = 0
        self.open_counter = 0
        self.blink_count = 0