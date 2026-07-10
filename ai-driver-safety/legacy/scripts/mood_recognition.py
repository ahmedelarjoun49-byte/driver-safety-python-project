from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


@dataclass(slots=True)
class EmotionConfig:

    model_path: str = "emotion_model.h5"

    input_size: int = 48


class EmotionDetector:
    """
    CNN based facial emotion detector.

    Returns emotion probabilities
    that can be fused with driver risk.
    """


    EMOTIONS = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprised",
    }


    def __init__(
        self,
        config: EmotionConfig | None = None,
    ) -> None:


        self.config = (
            config or EmotionConfig()
        )


        model_file = Path(
            self.config.model_path
        )


        if not model_file.exists():

            raise FileNotFoundError(
                f"Missing emotion model: {model_file}"
            )


        self.model = load_model(
            model_file
        )


        self.face_detector = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades
                +
                "haarcascade_frontalface_default.xml"
            )
        )



    def process(
        self,
        frame: np.ndarray,
    ) -> dict[str, float]:


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )


        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
        )


        if len(faces) == 0:

            return {
                "stress": 0.0,
                "surprise": 0.0,
            }



        x, y, w, h = faces[0]


        face = gray[
            y:y+h,
            x:x+w
        ]


        resized = cv2.resize(
            face,
            (
                self.config.input_size,
                self.config.input_size,
            ),
        )


        normalized = (
            resized.astype(
                "float32"
            )
            /
            255.0
        )


        tensor = np.expand_dims(
            normalized,
            axis=(0,-1),
        )


        prediction = (
            self.model.predict(
                tensor,
                verbose=0,
            )[0]
        )


        emotions = {
            self.EMOTIONS[index]:
            float(score)

            for index, score
            in enumerate(prediction)
        }



        return {

            # Emotional stress indicator
            "stress": round(
                max(
                    emotions["fearful"],
                    emotions["sad"],
                    emotions["angry"],
                ),
                3,
            ),


            # Surprise/distraction indicator
            "surprise": round(
                emotions["surprised"],
                3,
            ),

            "emotion": max(
                emotions,
                key=emotions.get,
            ),

        }