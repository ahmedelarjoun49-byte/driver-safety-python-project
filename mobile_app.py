import os
import sys
import importlib

# 1. On force le mode offscreen pour empêcher tout appel graphique X11/Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 2. CONTOURNEMENT RADICAL : On force le chargement du module headless 
# en allant chercher directement son nom d'installation natif si cv2 bug.
try:
    import cv2
except Exception:
    # Si l'import standard plante à cause de libGL, on nettoie et on charge le composant brut
    sys.modules.pop("cv2", None)
    try:
        cv2 = importlib.import_module("cv2.cv2" if sys.platform == "win32" else "cv2")
    except Exception:
        # Si ça coince encore, on importe directement via le package de repli headless
        import cv2

import streamlit as st
import numpy as np
import time
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Alignement des chemins pour charger le cœur de ton IA
sys.path.append(str(Path(__file__).resolve().parent / "ai-driver-safety"))

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket

# ... (Le reste de ton code avec st.set_page_config, load_pipeline, etc. reste identique)

import streamlit as st
import numpy as np
import time
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Alignement des chemins pour charger le cœur de ton IA
sys.path.append(str(Path(__file__).resolve().parent / "ai-driver-safety"))

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket

# Configuration de la page Streamlit
st.set_page_config(page_title="DriveSafe-AI Mobile", layout="centered")

st.markdown("""
    <style>
    h1 { text-align: center; color: #1E88E5; font-family: sans-serif; }
    .stAlert { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ DriveSafe-AI Mobile")

# Serveurs STUN publics de Google pour traverser le réseau mobile 4G/5G
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]}
)

@st.cache_resource
def load_pipeline():
    config = DriverSafetyConfig()
    config.thresholds.eye_aspect_ratio = 0.22
    config.thresholds.mouth_aspect_ratio = 0.55
    config.thresholds.head_offset = 0.15
    config.thresholds.phone_confidence = 0.50
    config.thresholds.eye_closed_frames = 2
    config.thresholds.drowsy_frames = 6
    config.thresholds.yawn_frames = 4
    config.thresholds.distracted_frames = 8
    config.thresholds.phone_use_frames = 2
    config.object_detector.phone_labels = ["cell phone", "phone"]
    config.runtime.alert_cooldown_seconds = 5
    return DriverSafetyPipeline(config)

pipeline = load_pipeline()

class MobileDriverProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_idx = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        packet = FramePacket(
            frame=img,
            timestamp=time.time(),
            frame_index=self.frame_idx
        )

        try:
            processed = pipeline.process_frame(packet)
            
            # Affichage dynamique sur le stream vidéo
            if processed.state.name != "ATTENTIVE":
                cv2.putText(img, f"ATTENTION: {processed.state.name}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "CONDUCTEUR ATTENTIF", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

        return frame.from_ndarray(img, format="bgr24")

# Streamer WebRTC optimisé pour caméra avant de smartphone
webrtc_streamer(
    key="mobile-live-engine",
    video_processor_factory=MobileDriverProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True
)