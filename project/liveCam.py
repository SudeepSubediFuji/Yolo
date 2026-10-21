import streamlit as st
from streamlit_webrtc import webrtc_streamer
# import cv2
import av
# import numpy as np
from ultralytics import YOLO


st.set_page_config(layout="wide",page_title="物体を検知するシステム")

model = YOLO("yolo26n.pt")

class VideoProcessor:
    def recv(self,frame):
        img = frame.to_ndarray(format = 'bgr24')
        # ,verbose=True
        ## tracker -> botsort.yaml -> better for detecting the person as it use reid to confirm the person
        ## bytetrack.yaml
        results = model.track(
            img,
            persist=True,
            tracker="tracktrack.yaml",
            verbose=True,
            # conf=0.20,          # Minimum detection confidence
            # iou=0.45,           # NMS IoU threshold
            )
        annotated_frame = results[0].plot()
        other = av.VideoFrame.from_ndarray(annotated_frame,format='bgr24')
        return other

st.write('Live Cam')
video = webrtc_streamer(
    key='ex',
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True,"audio":False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )