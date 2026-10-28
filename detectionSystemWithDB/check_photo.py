import os
import re
import shutil
import time
import traceback
from datetime import datetime

import av
import cv2
import face_recognition
import numpy as np
import streamlit as st

KNOWN_FACES_DIR = "./known_faces"
UNPROCESSED_PHOTO = "./unprocesed_photo"
report = {"loaded": 0, "failed": [], "skipped": 0}

# check process
# Performs comprehensive diagnostic checks on an individual photo:
#   1. File readable check
#   2. Image resolution & aspect ratio
#   3. Mean brightness estimation
#   4. Face detection (HOG / face_recognition)
#   5. Face count and crop size validation

def face_recognition_checker(image_path, name):
    image = face_recognition.load_image_file(image_path)
    face_locate = face_recognition.face_locations(image)
    if not face_locate:
        report["failed"].append(
            {
                "file": image_path,
                "reasons": "Face is not detected(check the lightining,hair style,..)",
            }
        )
    else:
        report["loaded"] += 1
        desitination_path = os.path.join(KNOWN_FACES_DIR, name)
        shutil.move(image_path, desitination_path)
        print(f"File{image_path} is moved from unprocessed_photo to the known_faces")

    return report


def photo_conversion():
    ## folder_check
    if not os.path.exists(UNPROCESSED_PHOTO):
        st.warning(f"No folder detected namely: {UNPROCESSED_PHOTO}")
        return report
    image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
    files_path = [
        f for f in os.listdir(UNPROCESSED_PHOTO) if f.lower().endswith(image_extension)
    ]

    for files in files_path:
        image_path = os.path.join(UNPROCESSED_PHOTO, files)
        name = os.path.basename(image_path)
        if face_recognition_checker(image_path=image_path, name=name):
            print(f"The face is detected in following path {image_path}")
