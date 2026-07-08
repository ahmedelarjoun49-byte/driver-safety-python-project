import os
import sys
import time
from pathlib import Path
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Configuration des chemins d'importation
root_path = Path(__file__).parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "ai-driver-safety"))

from driver_safety.config import DriverSafetyConfig
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.core.models import FramePacket

# Initialisation du pipeline d'IA
@st.cache_resource
def get_pipeline():
    return DriverSafetyPipeline(DriverSafetyConfig())

pipeline = get_pipeline()

st.title("DriveSafe-AI Dashboard")

# 1. Préparation de la mise en page (Layout à deux colonnes comme ton app locale)
col_video, col_metrics = st.columns([3, 1])

with col_metrics:
    st.subheader("Indicateurs de Risque")
    # Création des barres de progression vides qui se mettront à jour
    bar_eyes = st.progress(0.0, text="EYES CLOSED")
    bar_drowsy = st.progress(0.0, text="DROWSY")
    bar_yawn = st.progress(0.0, text="YAWNING")
    bar_distracted = st.progress(0.0, text="DISTRACTED")

class MobileDriverProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        packet = FramePacket(timestamp=time.time(), frame_index=0)
        
        try:
            # Traitement de l'image par ton pipeline
            processed = pipeline.process_frame(packet)
            
            # TODO: Si ton pipeline local stocke les scores dans 'processed', 
            # on les récupère ici pour mettre à jour l'UI Streamlit.
            
        except Exception:
            pass

        return frame.from_ndarray(img, format="bgr24")

with col_video:
    webrtc_streamer(
        key="driver-live-stream",
        video_processor_factory=MobileDriverProcessor,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True
    )