import os
import queue
import re
import shutil
import time
import traceback
from datetime import datetime

import av
import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO("yolo26n.pt")

# from app_utils import get_shared_runtime, snapshot_queue

# yolo_model, face_database, name_database = get_shared_runtime()

# import streamlit as st

# from app_utils import get_shared_runtime, snapshot_queue
snapshot_queue = queue.Queue(maxsize=50)

KNOWN_FACES_DIR = "./known_faces"
UNPROCESSED_FACES_DIR = "./unprocesed_photo"
UNPROCESSED_PHOTO_DIR = "./unprocesed_photo"
report = {"loaded": 0, "failed": [], "skipped": 0}

# check process
# Performs comprehensive diagnostic checks on an individual photo:
#   1. File readable check
#   2. Image resolution & aspect ratio
#   3. Mean brightness estimation
#   4. Face detection (HOG / face_recognition)
#   5. Face count and crop size validation


def photo_name_parse(path):
    base_name = os.path.splitext(path)[0]
    clean_name = re.sub(r"(_?\d+|_[a-zA-Z]+)$", "", base_name)
    return clean_name.strip()


def face_recognize_and_crop():

    # ※必要に応じて、以下の初期値変数を定義してください
    global last_unknown_capture_time
    global capture_cooldown_seconds
    last_unknown_capture_time = 0
    capture_cooldown_seconds = 5
    # name = "person_"
    # import queue
    # snapshot_queue = queue.Queue()

    # print(files_path)
    image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
    files_path = [
        f
        for f in os.listdir(UNPROCESSED_PHOTO_DIR)
        if f.lower().endswith(image_extension)
    ]

    for file in files_path:
        img_path = os.path.join(UNPROCESSED_PHOTO_DIR, file)
        # 画像をファイルパスから実際に読み込みます
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        results = yolo_model.track(img_path, tracker="bytetrack.yaml")[0]
        for result in results.boxes:
            if int(result.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                h, w, _ = img.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                person_crop = img[y1:y2, x1:x2]
                print(
                    f"person_crop data shape: {person_crop.shape if hasattr(person_crop, 'shape') else 'None'}"
                )
                if person_crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_crop)

                if face_locations:
                    now = time.time()
                    if (
                        "last_unknown_capture_time" in locals()
                        or "last_unknown_capture_time" in globals()
                    ):
                        pass
                    else:
                        last_unknown_capture_time = 0
                    if (
                        "capture_cooldown_seconds" in locals()
                        or "capture_cooldown_seconds" in globals()
                    ):
                        pass
                    else:
                        capture_cooldown_seconds = 5

                    if now - last_unknown_capture_time > capture_cooldown_seconds:
                        last_unknown_capture_time = now
                        top, right, bottom, left = face_locations[0]
                        face_crop = person_crop[top:bottom, left:right]
                        print(f"face_crop shape: {face_crop.shape}")
                        if face_crop.size > 0:
                            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
                            # person_name_list=file
                            # regex = "(_?\d+|_[a-zA-Z]+)$"
                            # filename = re.sub(regex,"",person_name_list)
                            # filename = f"{name if 'name' in locals() or 'name' in globals() else 'unknown_'}{timestamp_str}.jpg"
                            try:
                                # キューへの追加処理
                                # snapshot_queue.put_nowait((face_crop.copy(), filename))
                                # print(f"face crop added to queue.")

                                # 切り抜いた顔画像のみを processed_folder に直接保存します
                                save_path = os.path.join(KNOWN_FACES_DIR, file)
                                cv2.imwrite(save_path, face_crop)
                                print(
                                    f"Snapshot saved to processed folder: {save_path}"
                                )
                            except Exception as e:
                                print(f"Queue/Save error: {e}")


def photo_conversion():
    ## folder_check
    if not os.path.exists(UNPROCESSED_PHOTO):
        # st.warning(f"No folder detected namely: {UNPROCESSED_PHOTO}")
        print(f"No folder detected namely: {UNPROCESSED_PHOTO}")
        return report
    image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
    files_path = [
        f for f in os.listdir(UNPROCESSED_PHOTO) if f.lower().endswith(image_extension)
    ]

    for files in files_path:
        image_path = os.path.join(UNPROCESSED_PHOTO, files)
        name = os.path.basename(image_path)
        if face_recognition_checker(image_path, name=name):
            print(f" {report}")


# ------------Checking face_crop--------------
# image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
# files_path = [
#     f for f in os.listdir(UNPROCESSED_PHOTO) if f.lower().endswith(image_extension)
# ]
# for files in files_path:
#     image_path = os.path.join(UNPROCESSED_PHOTO, files)
#     print(f"image_path:{image_path}")
#     name = os.path.basename(image_path)
#     print(f"name check: {name}")
#     face_crop(image_path, name)

face_recognize_and_crop()
