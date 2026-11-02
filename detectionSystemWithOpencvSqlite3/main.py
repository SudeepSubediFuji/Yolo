import re
import time
from datetime import datetime
from email.mime import base

import cv2
import face_recognition
import numpy as np
import streamlit as st
from utils import (
    DISTANCE_THRESHOLD,
    UNKNOWN_DIR,
    get_shared_runtime,
    refresh_in_memory_db,
    snapshot_queue,
)
from database import KNOWN_FACES_DIR, scan_and_sync_folder
from torch import device

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📹 Live Face Detection & Identification")

# Acquire shared models, databases, and synchronization lock
yolo_model, face_database, name_database = get_shared_runtime()

# ----------------- SIDEBAR: DB MANAGEMENT & INSPECTION -----------------
st.sidebar.subheader("🗃️ Database & Sync Management")

if st.sidebar.button("🔄 Sync 'known_faces' Folder"):
    with st.spinner("Analyzing photos in folder..."):
        report = scan_and_sync_folder(KNOWN_FACES_DIR)
        print("scan and sync folder executed successfully!")
        refresh_in_memory_db()
        print("Refrest in memory db run successfully!")
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
        "Database is empty. Place photos in `unprocessed_photo/` and click Sync."
    )

# st.sidebar.subheader("Processed the unprocessed photo")

# if st.sidebar.button("🔄 Sync 'unprocessed' Folder"):
#     processed_photo()

start_button = st.button("ストリーミング開始", type="primary")

# ----------------- VIDEO PROCESSOR -----------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# scan_and_sync_folder(KNOWN_FACES_DIR)
UNPROCESSED_PHOTO_DIR = "./unprocesed_photo"


def videoProcessor(videocapture):
    global last_unknown_capture_time
    last_unknown_capture_time = 0.0
    global capture_cooldown_seconds
    capture_cooldown_seconds = 5.0
    video = st.empty()
    stop_button = st.button("ストリーミング停止", type="primary")

    if not videocapture.isOpened():
        st.warning("カメラを開けませんでした。")

    try:
        while videocapture.isOpened() and not stop_button:
            rt, frame = videocapture.read()
            if not rt == True:
                st.warning("フレームを取得できませんでした。")
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model.track(
                img,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
                device="cuda",
            )[0]
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # Coco dateset Class 0: Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    h, w, _ = img.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    person_crop = img[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_crop)

                    if face_locations:
                        current_encoding = face_recognition.face_encodings(
                            rgb_crop, face_locations
                        )[0]
                        # matched_name = "Unknown"
                        # box_color = (0, 0, 255)  # Red for unknown individuals

                        # 2. Acquire lock briefly to compare against in-memory DB
                        if len(face_database) > 0:
                            known_vectors = list(face_database.values())
                            known_embed_ids = list(face_database.keys())
                            # Calculate Euclidean distances to all registered embeddings
                            distances = face_recognition.face_distance(
                                known_vectors, current_encoding
                            )
                            best_match_idx = np.argmin(distances)
                            # If closest sample is below threshold, match found
                            if distances[best_match_idx] < DISTANCE_THRESHOLD:
                                best_id = known_embed_ids[best_match_idx]
                                matched_name = name_database[best_id]
                                # matched_name = "./known_faces/sudeep"
                                box_color = (0, 255, 0)  # Green for recognized person
                                display_person_name = re.sub(
                                    r"^.*/|(_?\d+|_[a-zA-Z]+)$", "", matched_name
                                )

                            else:
                                display_person_name = "Unknown"
                                box_color = (255, 0, 0)
                                now = time.time()
                                if (
                                    now - last_unknown_capture_time
                                    > capture_cooldown_seconds
                                ):
                                    last_unknown_capture_time = now
                                    top, right, bottom, left = face_locations[0]
                                    face_crop = person_crop[top:bottom, left:right]
                                    if face_crop.size > 0:
                                        timestamp_str = datetime.now().strftime(
                                            "%Y%m%d_%H%M%S"
                                        )
                                        filename = f"unknown_{timestamp_str}.jpg"
                                        try:
                                            snapshot_queue.put_nowait(
                                                (face_crop.copy(), filename)
                                            )
                                        except Exception:
                                            pass

                            # 3. Draw bounding box and label
                            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(
                                img,
                                display_person_name,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.85,
                                box_color,
                                2,
                            )
            # display_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.image(img, channels="RGB", width="stretch")

        # Render frame continuously outside detection loops
        # Convert BGR to RGB for browser presentation
        # display_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # video.image(img, channels="RGB", width="stretch")
        # video.image(img, channels="RGB", use_container_width=True)
    finally:
        cap.release()
        # cv2.destroyAllWindows()
        st.info("ストリーミングが停止され、カメラが解放されました。")


if start_button:
    videoProcessor(cap)
