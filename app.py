import os
import sys

# ==============================================================================
# 🛡️ PROTECTION ENVIRONNEMENT CLOUD (ANTI-LIBGL ERROR)
# ==============================================================================
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Si cv2 est déjà cassé dans l'environnement global, on nettoie préventivement
try:
    import cv2
except ImportError as e:
    if "libGL.so.1" in str(e):
        sys.modules.pop("cv2", None)
        # On force la recherche exclusive dans le headless
        import cv2

import streamlit as st
import numpy as np
import time

# Plus besoin de bidouiller sys.path ! Comme on est à la racine, 
# les imports natifs du projet fonctionnent directement.
from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Configuration globale de l'application
st.set_page_config(page_title="DriveSafe-AI Live", layout="centered")

st.markdown("""
    <style>
    h1 { text-align: center; color: #1E88E5; font-family: sans-serif; }
    .stAlert { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ DriveSafe-AI Live Monitor")

# Configuration des serveurs STUN de Google pour l'accès mobile (4G/5G)
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

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du pipeline IA : {e}")

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
            
            # Dessiner l'état d'alerte sur le flux vidéo retourné au téléphone
            if processed.state.name != "ATTENTIVE":
                cv2.putText(img, f"ATTENTION: {processed.state.name}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "CONDUCTEUR ATTENTIF", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

        return frame.from_ndarray(img, format="bgr24")

# Streamer d'envoi de la caméra
webrtc_streamer(
    key="driver-live-stream",
    video_processor_factory=MobileDriverProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True
)