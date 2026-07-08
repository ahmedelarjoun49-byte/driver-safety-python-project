import streamlit as st
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import numpy as np
import time
from pathlib import Path
import sys
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Alignement des chemins du projet
sys.path.append(str(Path(__file__).resolve().parent / "ai-driver-safety"))

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket

st.set_page_config(page_title="DriveSafe-AI Mobile", layout="centered")

st.markdown("""
    <style>
    h1 { text-align: center; color: #1E88E5; font-family: sans-serif; }
    .stAlert { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ DriveSafe-AI Mobile")

# Configuration des serveurs ICE de Google pour traverser les pare-feux mobiles
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
            
            # Dessiner l'état directement sur la vidéo de l'iPhone
            if processed.state.name != "ATTENTIVE":
                cv2.putText(img, f"ATTENTION: {processed.state.name}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "CONDUCTEUR ATTENTIF", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

        return frame.from_ndarray(img, format="bgr24")

# Lancement du streamer vidéo natif
webrtc_streamer(
    key="mobile-live-engine",
    video_processor_factory=MobileDriverProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    async_processing=True
)