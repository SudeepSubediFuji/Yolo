import os
import queue
import threading
import cv2

import streamlit as st
from db import init_db, load_verified_profiles_from_db
from ultralytics import YOLO

# Enforce thread execution limits before importing C++ vision libraries
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

DB_FILE = "faces.db"
UNKNOWN_DIR = "./unknown_captures"
os.makedirs(UNKNOWN_DIR, exist_ok=True)
# Matching distance threshold (0.45 = strict, 0.55 = balanced)
DISTANCE_THRESHOLD = 0.55

# Queue to handle disk writes off the video thread
snapshot_queue = queue.Queue(maxsize=50)


def snapshot_worker():
    """Background worker that writes JPEG files without blocking the video feed."""
    while True:
        task = snapshot_queue.get()
        if task is None:
            break
        image_crop, filename = task
        try:
            filepath = os.path.join(UNKNOWN_DIR, filename)
            ## cause of the freeze
            cv2.imwrite(filepath, image_crop)
            print(f"[AUDIT] Unknown face saved asynchronously: {filepath}")
        except Exception as e:
            print(f"[WORKER ERROR] Failed to save image: {e}")
        finally:
            snapshot_queue.task_done()


@st.cache_resource
def get_shared_runtime():
    """
    Initializes models and shared caches once per application session.
    """
    init_db()
    # Load YOLO26n nano model
    yolo_model = YOLO("yolo26n.pt")

    # Re-entrant lock for thread-safe access
    worker_thread = threading.Thread(target=snapshot_worker, daemon=True)
    worker_thread.start()
    ## initial db load
    # refresh_in_memory_db()
    # Read all verified face vectors from SQLite
    face_database, name_database = load_verified_profiles_from_db()

    return yolo_model, face_database, name_database


def refresh_in_memory_db():
    """Refreshes in-memory dictionaries directly from SQLite."""
    yolo_model, face_database, name_database = get_shared_runtime()
    new_faces, new_names = load_verified_profiles_from_db()
    face_database.clear()
    name_database.clear()
    face_database.update(new_faces)
    name_database.update(new_names)
