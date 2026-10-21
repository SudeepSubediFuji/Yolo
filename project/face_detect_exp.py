import os  
import threading
import face_recognition
import av  
import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)  
from ultralytics import YOLO 

st.title("Streamlit WebRTC Persistent Face Tracker")

SAVE_DIR = "./known_faces"
os.makedirs(SAVE_DIR, exist_ok=True)


# 2. Load Models
# Use Streamlit's caching decorator so the model stays in memory and doesn't reload on every app rerun
@st.cache_resource
def load_yolo_model():
    yolo_model = YOLO("yolov8n.pt")
    lock = threading.Lock()
    return yolo_model,lock

yolo_loaded_model,lock = load_yolo_model()

face_database = {}
name_database = {}

next_id = 1

DISTANCE_THRESHOLD = 0.5

# Pre-load known face images from folder at startup
for file in os.listdir(SAVE_DIR):
    if file.endswith((".jpg",".png")):
        ## ex "John" from "John.jpg"
        person_name = os.path.splitext(file)[0]
        image_path = os.path.join(SAVE_DIR,file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            face_database[next_id] = encoding[0]
            name_database[next_id] = person_name
            next_id +=1

class FaceTrackerProcessor(VideoProcessorBase):

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id  # Access the global ID counter variable inside this method

        img = frame.to_ndarray(format="bgr24")        # Pass the BGR image array to YOLO for object detection (verbose=False silences console logging)
        results = yolo_loaded_model(img, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                h, w, _ = img.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                person_crop = img[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_crop)

                if face_locations:
                    face_encoding = face_recognition.face_encodings(
                        rgb_crop, face_locations
                    )[0]

                    assigned_id = None

                    with lock:
                        if len(face_database) > 0:
                            known_encodings = list(face_database.values())
                            known_ids = list(face_database.keys())

                            distances = face_recognition.face_distance(
                                known_encodings, face_encoding
                            )
                            best_match_idx = np.argmin(distances)

                            if distances[best_match_idx] < DISTANCE_THRESHOLD:
                                assigned_id = known_ids[best_match_idx]

                        if assigned_id is None:
                            assigned_id = next_id
                            face_database[assigned_id] = face_encoding

                            top, right, bottom, left = face_locations[0]
                            face_crop = person_crop[top:bottom, left:right]

                            # If the face crop array is valid and non-empty, save it to disk
                            if face_crop.size > 0:
                                # Write the cropped face image as a JPEG file named with its assigned ID
                                cv2.imwrite(
                                    f"{SAVE_DIR}/Person_ID_{assigned_id}.jpg",
                                    face_crop,
                                )

                            # Increment the ID counter for the next new individual who enters the camera view
                            next_id += 1

                    person_name = name_database.get(assigned_id,f"ID: {assigned_id}")
                    # Draw a green bounding rectangle around the person on the main OpenCV frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Overlay the text showing the persistent ID above the person's bounding box
                    cv2.putText(
                        img,
                        f"{person_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_streamer(
    key="face-reid-tracker",  # Unique key identifier for Streamlit's component state
    video_processor_factory=FaceTrackerProcessor,  # Pass our custom processing class to handle incoming frames
    rtc_configuration=RTC_CONFIG,  # Apply network STUN configuration
    media_stream_constraints={
        "video": True,
        "audio": False,
    },  # Request camera access while disabling audio recording
)