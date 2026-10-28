import re

import av
import cv2
import face_recognition
import numpy as np
import streamlit as st
from app_utils import DISTANCE_THRESHOLD, get_shared_runtime, refresh_in_memory_db
from db import KNOWN_FACES_DIR, scan_and_sync_folder
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📹 Live Face Detection & Identification")

# Acquire shared models, databases, and synchronization lock
yolo_model, face_database, name_database, lock = get_shared_runtime()

# ----------------- SIDEBAR: DB MANAGEMENT & INSPECTION -----------------
st.sidebar.subheader("🗃️ Database & Sync Management")

if st.sidebar.button("🔄 Sync 'known_faces' Folder"):
    with st.spinner("Analyzing photos in folder..."):
        report = scan_and_sync_folder(KNOWN_FACES_DIR)
        refresh_in_memory_db()
        st.sidebar.success(
            f"Sync complete! Added: {report['loaded']}, Already Present: {report['skipped']}"
        )

        if report["failed"]:
            st.sidebar.error("⚠️ The following photos could NOT be detected:")
            for item in report["failed"]:
                st.sidebar.write(f"- `{item['file']}`: {item['reason']}")

st.sidebar.markdown("---")
st.sidebar.write(f"**Total Verified Vectors in Memory:** {len(face_database)}")

# Display unique names currently enrolled
if name_database:
    unique_names = sorted(list(set(name_database.values())))
    st.sidebar.write(f"**Registered Individuals ({len(unique_names)}):**")
    for name in unique_names:
        # Count how many photos are registered for this specific person
        count = sum(1 for n in name_database.values() if n == name)
        st.sidebar.write(f"- **{name}** ({count} verified sample photos)")
else:
    st.sidebar.warning(
        "Database is empty. Place photos in `known_faces/` and click Sync."
    )


# ----------------- VIDEO PROCESSOR -----------------
class RecognitionVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id

        img = frame.to_ndarray(format="bgr24")

        # Keep EVERYTHING strictly thread-locked inside WebRTC background cycles
        # with lock:
        results = yolo_model.track(
            img,
            verbose=False,
            persist=True,
            # conf=0.80,  # Minimum detection confidence
            # iou=0.45,
            tracker="bytetrack.yaml",
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
                    face_encoding = face_recognition.face_encodings(
                        rgb_crop, face_locations
                    )[0]
                    box_color = (0, 0, 255)  # Red for unknown
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
                                ## default_name save and check if found use the default name
                                default_name = f"Unknown_{assigned_id}"
                                name_database[assigned_id] = default_name
                                top, right, bottom, left = face_locations[0]
                                face_crop = person_crop[top:bottom, left:right]
                                if face_crop.size > 0:
                                    cv2.imwrite(
                                        f"{SAVE_DIR}/Unknown_{assigned_id}.jpg",
                                        face_crop,
                                    )
                                next_id += 1
                            person_name = name_database.get(
                                assigned_id, f"ID: {assigned_id}"
                            )
                            # regex = "(_?\d+|_[a-zA-Z]+)$"
                            # person_name = re.sub(regex,"",person_name_list)
                            # display_person_name = re.sub(
                            #     r"(_?\d+|_[a-zA-Z]+)$", "", person_name
                            # )
                            display_person_name = re.sub(
                                r"^.*/|(_?\d+|_[a-zA-Z]+)$", "", person_name
                            )
                            print(
                                f"[WARNING] \n**person_name:{display_person_name}**\n"
                            )
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                img,
                                f"{display_person_name}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------- WEBRTC STREAMER -----------------
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="verified-face-recognition",
    video_processor_factory=RecognitionVideoProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)
