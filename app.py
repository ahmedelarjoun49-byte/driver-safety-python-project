import sys
from pathlib import Path

sys.path.append(
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



st.set_page_config(
    page_title="DriveSafe AI",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ DriveSafe-AI Live Monitor")



RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers":[
            {
                "urls":[
                    "stun:stun.l.google.com:19302"
                ]
            }
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



class DriverProcessor(VideoProcessorBase):


    def __init__(self):

        self.status = "INITIALISATION"
        self.frame_index = 0



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

            self.status = result.state.name


        except Exception as e:

            self.status = "ERROR"



        return frame





ctx = webrtc_streamer(

    key="driver-camera",

    video_processor_factory=DriverProcessor,

    rtc_configuration=RTC_CONFIGURATION,


    media_stream_constraints={

        "video":{
            "facingMode":"user"
        },

        "audio":False
    },


    async_processing=False

)



status = st.empty()



if ctx.video_processor:


    current = ctx.video_processor.status



    if current == "ATTENTIVE":

        status.success(
            "✅ Conducteur attentif"
        )


    elif current == "ERROR":

        status.error(
            "❌ Erreur IA"
        )


    else:

        status.warning(
            f"🚨 {current}"
        )


else:

    status.info(
        "📷 Activation caméra..."
    )