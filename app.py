import os
import sys
import time
import streamlit as st
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Imports directs de ton pipeline d'IA
from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket

st.set_page_config(page_title="DriveSafe-AI Mobile", layout="centered")
st.title("🛡️ DriveSafe-AI Live Monitor")

# Configuration des serveurs de connexion pour le réseau mobile (4G/5G)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_pipeline():
    config = DriverSafetyConfig()
    config.thresholds.eye_aspect_ratio = 0.22
    config.thresholds.mouth_aspect_ratio = 0.55
    config.thresholds.head_offset = 0.15
    config.thresholds.phone_confidence = 0.50
    return DriverSafetyPipeline(config)

pipeline = load_pipeline()

# Initialisation de l'état dans la session Streamlit
if "driver_status" not in st.session_state:
    st.session_state.driver_status = "INITIALISATION"

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
            # Ton IA analyse l'image ici
            processed = pipeline.process_frame(packet)
            st.session_state.driver_status = processed.state.name
        except Exception:
            pass

        return frame

# Zone d'affichage du statut sur l'écran du téléphone
status_placeholder = st.empty()

if st.session_state.driver_status != "ATTENTIVE":
    status_placeholder.error(f"🚨 ALERTE : {st.session_state.driver_status}")
else:
    status_placeholder.success("✅ CONDUCTEUR ATTENTIF")

# Lancement du flux de la caméra frontale de l'iPhone
webrtc_streamer(
    key="driver-live-stream",
    video_processor_factory=MobileDriverProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": True},
    async_processing=True
)