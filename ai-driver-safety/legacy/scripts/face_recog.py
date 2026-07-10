from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import face_recognition
import numpy as np


@dataclass(slots=True)
class FaceRecognitionConfig:

    tolerance: float = 0.6

    model: str = "hog"

    known_faces: dict[str, str] = field(
        default_factory=dict
    )



class FaceRecognitionDetector:
    """
    Detects faces and optionally identifies the driver.

    Output:
    - face_detected
    - identity confidence
    - bounding box
    """


    def __init__(
        self,
        config: FaceRecognitionConfig | None = None,
    ) -> None:


        self.config = (
            config or FaceRecognitionConfig()
        )


        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []


        self._load_known_faces()



    def _load_known_faces(self) -> None:

        for name, image_path in (
            self.config.known_faces.items()
        ):

            path = Path(image_path)


            if not path.exists():
                continue


            image = (
                face_recognition
                .load_image_file(path)
            )


            encodings = (
                face_recognition
                .face_encodings(image)
            )


            if encodings:

                self.known_encodings.append(
                    encodings[0]
                )

                self.known_names.append(
                    name
                )



    def process(
        self,
        frame: np.ndarray,
    ) -> dict:


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )


        locations = (
            face_recognition
            .face_locations(
                rgb,
                model=self.config.model,
            )
        )


        encodings = (
            face_recognition
            .face_encodings(
                rgb,
                locations,
            )
        )


        results = []


        for location, encoding in zip(
            locations,
            encodings,
        ):

            top, right, bottom, left = location


            identity = "Unknown"

            confidence = 0.0


            if self.known_encodings:

                distances = (
                    face_recognition
                    .face_distance(
                        self.known_encodings,
                        encoding,
                    )
                )


                best_index = int(
                    np.argmin(distances)
                )


                distance = distances[
                    best_index
                ]


                if distance < self.config.tolerance:

                    identity = (
                        self.known_names[
                            best_index
                        ]
                    )


                    confidence = round(
                        1 - distance,
                        3,
                    )


            results.append(
                {
                    "identity": identity,

                    "confidence": confidence,

                    "bbox": (
                        left,
                        top,
                        right-left,
                        bottom-top,
                    ),
                }
            )



        return {

            "face_detected": (
                len(results) > 0
            ),

            "faces": results,

            "face_count": len(results),
        }