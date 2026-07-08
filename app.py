import sys
import time
from pathlib import Path

# Add the local package to Python's search path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ai-driver-safety"))

import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket


st.set_page_config(page_title="DriveSafe-AI Mobile", layout="centered")
st.title("🛡️ DriveSafe-AI Live Monitor")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
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
            frame_index=self.frame_idx,
        )

        try:
            processed = pipeline.process_frame(packet)
            st.session_state.driver_status = processed.state.name
        except Exception as e:
            print(e)

        return frame


status_placeholder = st.empty()

if st.session_state.driver_status == "ATTENTIVE":
    status_placeholder.success("✅ CONDUCTEUR ATTENTIF")
else:
    status_placeholder.error(f"🚨 ALERTE : {st.session_state.driver_status}")


webrtc_streamer(
    key="driver-live-stream",
    video_processor_factory=MobileDriverProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False,
    },
    async_processing=True,
)