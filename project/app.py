import streamlit as st
from PIL import Image
import os
import tempfile
import cv2
# import torch
import numpy as np
from ultralytics import YOLO


st.set_page_config(layout="wide",page_title="物体を検知するシステム")

model = YOLO("yolo26n.pt")

## ,disabled=not enable
picture = st.camera_input(label="Campera input",resolution="1080p",on_change="rerun")

# if picture:
#     st.image(picture)
#     bytes_data = picture.getvalue()
#     ##torchvision.io.decode_image(input: Union[Tensor, str], mode: ImageReadMode = ImageReadMode.UNCHANGED, apply_exif_orientation: bool = False) → Tensor
#     ## mode : 3 -> meaning 3d
#     # torch_img = torch.ops.image.decode_image(
#     #     torch.from_numpy(np.frombuffer(bytes_data,np.uint8)),3
#     #     )
#     # print(bytes_data)
#     cv2_img = cv2.imdecode(np.frombuffer(bytes_data,np.uint8),cv2.IMREAD_COLOR)

#     # checking the type of the data
#     st.write(type(cv2_img))
#     # shape
#     st.write(cv2_img.shape)

## numpy to extract type and value of img
if picture:
    st.image(picture)
    img = Image.open(picture)
    frame = np.array(img)
    # st.write(type(frame))
    st.write(frame.shape)
    results = model(frame)
    ## BBOX creation
    annotated_frame = results[0].plot()

    # Display result
    st.image(annotated_frame,channels="RGB")
    ##,captions="Detected Objects"


    
