import os  # Provides functions for interacting with the operating system (e.g., creating directories)
import threading  # Enables thread locking so shared variables can be updated safely across concurrent video frames
import av  # PyAV library used by streamlit-webrtc to convert video frames into NumPy arrays
import cv2  # OpenCV library for image processing, drawing bounding boxes, and saving image files
import face_recognition  # Library for detecting face locations and generating 128-dimensional face embeddings
import numpy as np  # Numerical Python library for matrix operations and finding array minima
import streamlit as st  # Main web application framework for rendering the UI
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)  # Components for WebRTC streaming in Streamlit
from ultralytics import YOLO  # Ultralytics framework for loading and running YOLO detection models

# Set the title displayed at the top of the Streamlit web application
st.title("Streamlit WebRTC Persistent Face Tracker")

# 1. Directory Setup for Saved Crop Images
# Define the local folder path where newly recognized face cropped images will be saved
SAVE_DIR = "./known_faces"
# Create the directory if it does not already exist on disk
os.makedirs(SAVE_DIR, exist_ok=True)


# 2. Load Models
# Use Streamlit's caching decorator so the model stays in memory and doesn't reload on every app rerun
@st.cache_resource
def load_yolo_model():
    # Load the lightweight pre-trained YOLOv8 nano model
    return YOLO("yolov8n.pt")


# Execute the cached function to retrieve the loaded YOLO model instance
yolo_model = load_yolo_model()

# Create a Threading Lock to prevent race conditions when reading/writing global memory from video threads
lock = threading.Lock()

# Initialize an in-memory dictionary to store registered face embeddings: { assigned_id: embedding_vector }
face_database = {}

# Set the starting integer counter for assigning persistent IDs to newly detected individuals
next_id = 1

# Threshold for face distance matching (lower value = stricter match; values around 0.4 to 0.6 are typical)
DISTANCE_THRESHOLD = 0.5


# 3. WebRTC Frame Processor
# Define a class inheriting from VideoProcessorBase to intercept and process each live WebRTC video frame
class FaceTrackerProcessor(VideoProcessorBase):

    # The recv callback method executes automatically for every incoming frame in the video stream
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global next_id  # Access the global ID counter variable inside this method

        # Convert the incoming PyAV VideoFrame object into a 3D NumPy array formatted as BGR for OpenCV
        img = frame.to_ndarray(format="bgr24")

        # Pass the BGR image array to YOLO for object detection (verbose=False silences console logging)
        results = yolo_model(img, verbose=False)[0]

        # Loop through each bounding box detected in the current frame by YOLO
        for box in results.boxes:
            # Check if the detected object belongs to Class 0 (which corresponds to 'Person' in COCO dataset)
            if int(box.cls[0]) == 0:
                # Extract coordinates [x1, y1, x2, y2] from tensor and convert them to integers
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Get image dimensions (height, width, channels) to ensure coordinates stay inside frame limits
                h, w, _ = img.shape
                # Clamp coordinates to ensure they do not exceed image boundaries (prevents slicing crashes)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # Crop the region corresponding to the detected person from the frame
                person_crop = img[y1:y2, x1:x2]
                # If the cropped region is empty (0 size), skip further processing for this box
                if person_crop.size == 0:
                    continue

                # Convert the cropped image from OpenCV BGR color format to RGB format required by face_recognition
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                # Detect bounding box locations of any human faces present inside the cropped person region
                face_locations = face_recognition.face_locations(rgb_crop)

                # If at least one face was located inside the person's bounding box
                if face_locations:
                    # Compute the 128-dimensional numerical feature vector (embedding) for the first face found
                    face_encoding = face_recognition.face_encodings(
                        rgb_crop, face_locations
                    )[0]

                    # Initialize assigned_id as None to indicate the face is unverified so far
                    assigned_id = None

                    # Acquire the thread lock so no other frame callback alters face_database simultaneously
                    with lock:
                        # If we already have registered face encodings saved in memory
                        if len(face_database) > 0:
                            # Extract all saved feature vectors into a list
                            known_encodings = list(face_database.values())
                            # Extract all corresponding persistent ID numbers into a list
                            known_ids = list(face_database.keys())

                            # Calculate Euclidean distances between the current face vector and all saved vectors
                            distances = face_recognition.face_distance(
                                known_encodings, face_encoding
                            )
                            # Identify the index position corresponding to the lowest distance value
                            best_match_idx = np.argmin(distances)

                            # If the lowest distance is smaller than our threshold, treat it as a recognized face
                            if distances[best_match_idx] < DISTANCE_THRESHOLD:
                                # Re-assign the previously saved persistent ID to this person
                                assigned_id = known_ids[best_match_idx]

                        # If no matching face was found in the database (or database is empty), register as new
                        if assigned_id is None:
                            # Assign the current value of next_id as this person's new persistent ID
                            assigned_id = next_id
                            # Store the new face embedding vector in the dictionary using the assigned ID as key
                            face_database[assigned_id] = face_encoding

                            # Extract face corner coordinates (top, right, bottom, left) from face_locations
                            top, right, bottom, left = face_locations[0]
                            # Crop just the face area from the person crop
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

                    # Draw a green bounding rectangle around the person on the main OpenCV frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Overlay the text showing the persistent ID above the person's bounding box
                    cv2.putText(
                        img,
                        f"ID: {assigned_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

        # Re-encapsulate the annotated OpenCV NumPy image array back into a PyAV VideoFrame to send back to WebRTC
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# 4. Streamlit WebRTC Streamer
# Configure Google STUN servers to handle WebRTC network connection traversal across local networks
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Instantiate the Streamlit WebRTC video player widget in the browser UI
webrtc_streamer(
    key="face-reid-tracker",  # Unique key identifier for Streamlit's component state
    video_processor_factory=FaceTrackerProcessor,  # Pass our custom processing class to handle incoming frames
    rtc_configuration=RTC_CONFIG,  # Apply network STUN configuration
    media_stream_constraints={
        "video": True,
        "audio": False,
    },  # Request camera access while disabling audio recording
)