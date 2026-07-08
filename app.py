import time
import threading
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket


# -----------------------------
# Streamlit configuration
# -----------------------------

st.set_page_config(
    page_title="DriveSafe-AI Mobile",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ DriveSafe-AI Live Monitor")


# -----------------------------
# WebRTC configuration
# -----------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# -----------------------------
# Load AI model once
# -----------------------------

@st.cache_resource
def load_pipeline():

    config = DriverSafetyConfig()

    config.thresholds.eye_aspect_ratio = 0.22
    config.thresholds.mouth_aspect_ratio = 0.55
    config.thresholds.head_offset = 0.15
    config.thresholds.phone_confidence = 0.50

    return DriverSafetyPipeline(config)


pipeline = load_pipeline()


# -----------------------------
# Status storage
# Thread-safe
# -----------------------------

if "status" not in st.session_state:
    st.session_state.status = "INITIALISATION"


status_lock = threading.Lock()


# -----------------------------
# Video processor
# -----------------------------

class MobileDriverProcessor(VideoProcessorBase):

    def __init__(self):

        self.frame_index = 0
        self.current_status = "INITIALISATION"


    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_index += 1


        packet = FramePacket(
            frame=img,
            timestamp=time.time(),
            frame_index=self.frame_index
        )


        try:

            result = pipeline.process_frame(packet)

            self.current_status = result.state.name


        except Exception as e:

            self.current_status = "ERROR"


        return frame



# -----------------------------
# Camera
# -----------------------------

ctx = webrtc_streamer(
    key="driver-camera",

    video_processor_factory=MobileDriverProcessor,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": {
            "facingMode": "user"
        },
        "audio": False
    },

    async_processing=False
)



# -----------------------------
# Display status
# -----------------------------

status_box = st.empty()


if ctx.video_processor:

    current = ctx.video_processor.current_status


    if current != "ATTENTIVE":

        status_box.error(
            f"🚨 ALERTE : {current}"
        )

    else:

        status_box.success(
            "✅ CONDUCTEUR ATTENTIF"
        )

else:

    status_box.info(
        "📷 Activation caméra..."
    )