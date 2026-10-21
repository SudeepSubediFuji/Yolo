import os  
# CRITICAL FIX 1: Prevent OpenBLAS/dlib from colliding with Streamlit's multi-threading pool
# These variables MUST be defined BEFORE importing face_recognition or cv2!
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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

# 1. Load Models & Global Lock via cache_resource
@st.cache_resource
def load_yolo_model():
    yolo_model = YOLO("yolo26n.pt")
    lock = threading.Lock()
    return yolo_model, lock

yolo_loaded_model, lock = load_yolo_model()

# CRITICAL FIX 2: Isolate the database loading into a cached resource
# This prevents it from conflicting or re-running unsafely during active streams.
@st.cache_resource
def load_initial_database(save_dir):
    face_db = {}
    name_db = {}
    current_id = 1
    
    if not os.path.exists(save_dir):
        return face_db, name_db, current_id

    for file in os.listdir(save_dir):
        if file.endswith((".jpg", ".png")):
            person_name = os.path.splitext(file)[0]
            image_path = os.path.join(save_dir, file)
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    # CRITICAL FIX 3: Unpack the first item [0] to keep the shape consistent
                    face_db[current_id] = encoding[0]
                    name_db[current_id] = person_name
                    current_id += 1
            except Exception as e:
                print(f"Error reading startup file {file}: {e}")
                
    return face_db, name_db, current_id

# Initialize local structures safely
face_database, name_database, next_id = load_initial_database(SAVE_DIR)
DISTANCE_THRESHOLD = 0.5

class FaceTrackerProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id  

        img = frame.to_ndarray(format="bgr24")        
        
        # Keep EVERYTHING strictly thread-locked inside WebRTC background cycles
        with lock:
            results = yolo_loaded_model.track(
                img,
                verbose=False,
                persist=True,
                tracker="tracktrack.yaml",
                )[0]
            
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
                        # Extract single array element safely
                        face_encoding = face_recognition.face_encodings(rgb_crop, face_locations)[0]
                        assigned_id = None

                        if len(face_database) > 0:
                            known_encodings = list(face_database.values())
                            known_ids = list(face_database.keys())

                            distances = face_recognition.face_distance(known_encodings, face_encoding)
                            best_match_idx = np.argmin(distances)

                            if distances[best_match_idx] < DISTANCE_THRESHOLD:
                                assigned_id = known_ids[best_match_idx]

                        if assigned_id is None:
                            assigned_id = next_id
                            face_database[assigned_id] = face_encoding

                            ## default_name save and check if found use the default name
                            default_name = f"Person_ID_{assigned_id}"
                            name_database[assigned_id]= default_name

                            top, right, bottom, left = face_locations[0]
                            face_crop = person_crop[top:bottom, left:right]

                            if face_crop.size > 0:
                                cv2.imwrite(
                                    f"{SAVE_DIR}/Person_ID_{assigned_id}.jpg",
                                    face_crop,
                                )
                            next_id += 1

                        person_name = name_database.get(assigned_id, f"ID: {assigned_id}")
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
    key="face-reid-tracker",  
    video_processor_factory=FaceTrackerProcessor,  
    rtc_configuration=RTC_CONFIG,  
    media_stream_constraints={
        "video": True,
        "audio": False,
    },  
)

# ==========================================
# ADD THIS DEBUG BLOCK TO INSPECT YOUR DATA
# ==========================================
st.sidebar.subheader("🗄️ Loaded Database Registry")
st.sidebar.write(f"Total Registered Profiles: {len(name_database)}")

if len(name_database) > 0:
    # 1. Inspect the human-readable names mapped to IDs
    st.sidebar.write("**Active Names Mapping:**", name_database)
    
    # 2. Inspect the raw vector structure (Checking dimensions)
    sample_id = list(face_database.keys())[0]
    sample_vector = face_database[sample_id]
    st.sidebar.write(f"**Vector Dimensions for ID {sample_id}:**", sample_vector.shape)
else:
    st.sidebar.error("⚠️ No profiles loaded. Ensure 'known_faces' has valid images!")

# st.sidebar.write(f"Name database Content", name_database)
# name_id = list(name_database.keys())[0]
# name_vector = name_database[sample_id]
# # st.sidebar.write(f"Name database Content", name_db)
# # st.sidebar.write(f"Name database Content", face_db)




