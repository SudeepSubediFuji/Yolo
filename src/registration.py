import av  
import cv2
import os
import streamlit as st
import numpy as np
import re
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)  
from ultralytics import YOLO 

from utill import load_initial_database


st.title("登録画面")
SAVE_DIR = "./new_face"
os.makedirs(SAVE_DIR, exist_ok=True)

face_database, name_database, next_id = load_initial_database(SAVE_DIR)

person_name_registraion = st.text_input("お名前を登録してください")
if person_name_registraion:
    st.write("Name has been registrated")
    print(f"[INFO] Registered name:{person_name_registraion}")

# Initialize local structures safely
# face_database, name_database, next_id = load_initial_database(SAVE_DIR)
DISTANCE_THRESHOLD = 0.5

class FaceTrackerProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id  

        img = frame.to_ndarray(format="bgr24")        
        
        # Keep EVERYTHING strictly thread-locked inside WebRTC background cycles
        with lock:
            # results = yolo_loaded_model.track(
            #     img,
            #     verbose=False,
            #     persist=True,
            #     conf=0.80,          # Minimum detection confidence
            #     iou=0.45, 
            #     tracker="tracktrack.yaml",
            #     )[0]
            results = yolo_loaded_model.track(
                img,
                verbose=False,
                persist=True,
                conf=0.80,          # Minimum detection confidence
                iou=0.45, 
                tracker="tracktrack.yaml",
                # imgsz=1080
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
                            default_name = f"{person_name_registraion}_{assigned_id}"
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
                        # regex = "(_?\d+|_[a-zA-Z]+)$"
                        # person_name = re.sub(regex,"",person_name_list)
                        print(f"[WARNING] \n**person_name:{person_name}**\n")
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
# ★ UPDATED SIDEBAR WITH CACHE CLEAR BUTTON
# ==========================================
st.sidebar.subheader("🗄️ Database Registry Inspection")

# Reload button to purge Streamlit cache when you drop new images into known_faces/
if st.sidebar.button("🔄 Reload 'known_faces' Folder"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.write(f"**Total Loaded Profiles:** {len(name_database)}")

if len(name_database) > 0:
    st.sidebar.markdown("---")
    st.sidebar.write("### 👤 Loaded Names:")
    for id_num, name in name_database.items():
        st.sidebar.text(f"ID {id_num}: {name}")
else:
    st.sidebar.error(
        "⚠️ No profiles loaded! Check terminal logs or ensure 'sudh.jpg' has a clear front-facing face."
    )
# Checklist to verify sudh.jpg:




