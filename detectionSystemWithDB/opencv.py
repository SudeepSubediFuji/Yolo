import os

import cv2
import streamlit as st

##ffmpeg and
# System install package :: v4l-utils

# ls -l /dev/video* -> find the video source
# ls -l /dev/video*
# crw-rw----+ 1 root video 81, 0 10月 28 10:03 /dev/video0
# crw-rw----+ 1 root video 81, 1 10月 28 10:03 /dev/video1


st.title("Opencv")
# cam = st.columns(1)
video = st.empty()
# cap = cv2.VideoCapture(-1, cv2.CAP_V4L) cv2.CAP_DSHOW
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)


cap1 = cv2.VideoCapture(1, cv2.CAP_V4L)
# cap2 = cv2.VideoCapture(2)
# cv2


def opencvRTC(videocapture):
    while videocapture.isOpened():
        rt, frame = videocapture.read()
        if rt == True:
            cv2.imshow("WebcamLive", frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.image(frame, channels="RGB")


opencvRTC(cap)
# opencvRTC(cap1)
# opencvRTC(cap2)
cap.release()
cv2.destroyAllWindows()

## OpenCv camera data obtanining process failed in this steps.
