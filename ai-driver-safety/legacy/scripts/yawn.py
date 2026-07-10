from __future__ import annotations

from dataclasses import dataclass

import cv2
import dlib
import numpy as np



@dataclass(slots=True)
class YawnConfig:

    predictor_path: str = (
        "shape_predictor_68_face_landmarks.dat"
    )

    mouth_ratio_threshold: float = 0.55

    yawn_frames_required: int = 8



class YawnDetector:
    """
    Detects yawning using facial landmarks.

    Uses Mouth Aspect Ratio (MAR)
    instead of fixed pixel distance.
    """


    def __init__(
        self,
        config: YawnConfig | None = None,
    ) -> None:


        self.config = (
            config or YawnConfig()
        )


        self.detector = (
            dlib.get_frontal_face_detector()
        )


        self.predictor = (
            dlib.shape_predictor(
                self.config.predictor_path
            )
        )


        self.yawn_counter = 0
        self.total_yawns = 0



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

                "yawning": 0.0,

                "mouth_open_ratio": 0.0,

                "yawn_count": float(
                    self.total_yawns
                ),
            }



        landmarks = self.predictor(
            gray,
            faces[0],
        )


        points = np.array(
            [
                (
                    p.x,
                    p.y,
                )

                for p in landmarks.parts()
            ]
        )


        mar = self._mouth_aspect_ratio(
            points
        )


        yawning = 0.0


        if mar > self.config.mouth_ratio_threshold:

            self.yawn_counter += 1


            yawning = min(
                self.yawn_counter /
                self.config.yawn_frames_required,
                1.0,
            )


        else:

            if (
                self.yawn_counter >=
                self.config.yawn_frames_required
            ):

                self.total_yawns += 1


            self.yawn_counter = 0



        return {

            "yawning": round(
                yawning,
                3,
            ),

            "mouth_open_ratio": round(
                mar,
                3,
            ),

            "yawn_count": float(
                self.total_yawns
            ),
        }



    @staticmethod
    def _mouth_aspect_ratio(
        landmarks: np.ndarray,
    ) -> float:

        """
        Calculates Mouth Aspect Ratio (MAR).

        Higher value = mouth more open.
        """


        # Inner mouth landmarks

        top = np.linalg.norm(
            landmarks[62]
            -
            landmarks[66]
        )


        horizontal = np.linalg.norm(
            landmarks[60]
            -
            landmarks[64]
        )


        if horizontal == 0:

            return 0.0


        return float(
            top /
            horizontal
        )



    def reset(self) -> None:

        self.yawn_counter = 0
        self.total_yawns = 0