import os  
# CRITICAL FIX 1: Prevent thread collisions before dlib/cv2 imports
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

# 1. Load YOLO Model and Thread Lock
@st.cache_resource
def load_yolo_model():
    # ★ UPDATED: Fixed weight file name to official 'yolov8n.pt'
    yolo_model = YOLO("yolov8n.pt")
    lock = threading.Lock()
    return yolo_model, lock

yolo_loaded_model, lock = load_yolo_model()

# 2. Load Startup Database from known_faces Folder
@st.cache_resource
def load_initial_database(save_dir):
    face_db = {}
    name_db = {}
    current_id = 1
    
    if not os.path.exists(save_dir):
        return face_db, name_db, current_id

    for file in os.listdir(save_dir):
        # ★ UPDATED: Added support for .jpeg files along with .jpg and .png
        if file.endswith((".jpg", ".jpeg", ".png")):
            person_name = os.path.splitext(file)[0]  # "Sam.jpg" -> "Sam"
            image_path = os.path.join(save_dir, file)
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    face_db[current_id] = encoding[0]
                    name_db[current_id] = person_name
                    print(f"[LOADED] ID {current_id}: {person_name}")
                    current_id += 1
            except Exception as e:
                print(f"Error reading startup file {file}: {e}")
                
    return face_db, name_db, current_id

# Initialize databases
face_database, name_database, next_id = load_initial_database(SAVE_DIR)

# ★ UPDATED: Raised distance threshold to 0.55 to prevent false new registrations under different lighting
DISTANCE_THRESHOLD = 0.55 

# 3. WebRTC Frame Processor
class FaceTrackerProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id  

        img = frame.to_ndarray(format="bgr24")        
        
        with lock:
            # ★ UPDATED: Changed tracker config to standard 'bytetrack.yaml'
            results = yolo_loaded_model.track(
                img,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )[0]
            
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # Class 0: Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w, _ = img.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    person_crop = img[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_crop)

                    if face_locations:
                        face_encoding = face_recognition.face_encodings(rgb_crop, face_locations)[0]
                        assigned_id = None

                        # Match face against existing database
                        if len(face_database) > 0:
                            known_encodings = list(face_database.values())
                            known_ids = list(face_database.keys())

                            distances = face_recognition.face_distance(known_encodings, face_encoding)
                            best_match_idx = np.argmin(distances)

                            if distances[best_match_idx] < DISTANCE_THRESHOLD:
                                assigned_id = known_ids[best_match_idx]

                        # Register new face if unmatched
                        if assigned_id is None:
                            assigned_id = next_id
                            face_database[assigned_id] = face_encoding
                            
                            # ★ UPDATED: Explicitly set name_database[assigned_id] for newly discovered people
                            default_name = f"Person_ID_{assigned_id}"
                            name_database[assigned_id] = default_name

                            top, right, bottom, left = face_locations[0]
                            face_crop = person_crop[top:bottom, left:right]

                            if face_crop.size > 0:
                                cv2.imwrite(
                                    f"{SAVE_DIR}/{default_name}.jpg",
                                    face_crop,
                                )
                            next_id += 1

                        # Get display name from name_database
                        person_name = name_database.get(assigned_id, f"ID: {assigned_id}")
                        
                        # Render Bounding Box and Name
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

# 4. WebRTC Widget
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

# ★ UPDATED: Cleaned and safe sidebar display block (removes out-of-scope variable references and index errors)
st.sidebar.subheader("🗄️ Database Registry Inspection")
st.sidebar.write(f"**Total Registered Profiles:** {len(name_database)}")

if len(name_database) > 0:
    st.sidebar.markdown("---")
    st.sidebar.write("### 👤 Name Mapping (`name_database`)")
    for id_num, name in name_database.items():
        st.sidebar.text(f"ID {id_num}: {name}")

    st.sidebar.markdown("---")
    st.sidebar.write("### 🧬 Vector Storage (`face_database`)")
    sample_id = list(face_database.keys())[0]
    sample_vector = face_database[sample_id]
    st.sidebar.write(f"**ID {sample_id} Embedding Shape:** `{sample_vector.shape}` (128-D Vector)")
else:
    st.sidebar.warning("⚠️ No profiles loaded. Ensure 'known_faces' has 'Sam.jpg'!")



if face_recognition.compare_faces(knoown_face_encoding,face_encoding)