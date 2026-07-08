import sys
from pathlib import Path

# Add AI package path
sys.path.insert(
    0,
    str(Path(__file__).parent / "ai-driver-safety")
)


import time
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
# Streamlit config
# -----------------------------

st.set_page_config(
    page_title="DriveSafe-AI",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ DriveSafe-AI Live Monitor")



# -----------------------------
# WebRTC
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
# Load AI pipeline
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
# Video processor
# -----------------------------

class DriverProcessor(VideoProcessorBase):

    def __init__(self):

        self.status = "INITIALISATION"
        self.frame_index = 0



    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        self.frame_index += 1


        packet = FramePacket(
            frame=image,
            timestamp=time.time(),
            frame_index=self.frame_index
        )


        try:

            result = pipeline.process_frame(packet)

            self.status = result.state.name


        except Exception as e:

            self.status = "ERROR"


        return frame





# -----------------------------
# Camera
# -----------------------------

ctx = webrtc_streamer(

    key="driver-camera",

    video_processor_factory=DriverProcessor,

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
# Status
# -----------------------------

status_box = st.empty()



if ctx.video_processor:

    status = ctx.video_processor.status


    if status == "ATTENTIVE":

        status_box.success(
            "✅ Conducteur attentif"
        )


    elif status == "ERROR":

        status_box.error(
            "❌ Erreur IA"
        )


    else:

        status_box.warning(
            f"🚨 {status}"
        )


else:

    status_box.info(
        "📷 Activation caméra..."
    )