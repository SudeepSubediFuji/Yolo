import glob
import os 
import cv2
import numpy as np
from pathlib import Path

REAL_FACE_DIR = "./../images/train/*.jpg"
OUTPUT_SKETCH = "./false_data/train/"
OUTPUT_SCANLINE = "./false_data/train/"
OUTPUT_CUTOUT = "./false_data/train/"

LABEL_PATH = "./false_data/labels"
os.makedirs(LABEL_PATH,exist_ok=True)

def img_to_sketch():
    os.makedirs(OUTPUT_SKETCH,exist_ok=True)

    face_files = glob.glob(REAL_FACE_DIR)[:399]
    
    for idx,img_path in enumerate(face_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        ## Convert to sketch
        blurred = cv2.GaussianBlur(inverted,(21,21),sigmaX=0,sigmaY=0)
        sketch = cv2.divide(gray,255-blurred,scale=256)
    
        ## save sketch 
        cv2.imwrite(f"{OUTPUT_SKETCH}/synthetic_sketch_{idx:04d}.jpg",sketch)
        p = Path(f"{LABEL_PATH}/synthetic_sketch_{idx:04d}.txt")
        p.touch()
        print("Write process completed.")
    print("Converted to sketch successfully!")

# def convert_to_cutout_print(img:np.array)-> np.array:
def convert_to_screen_glare():
    os.makedirs(OUTPUT_SCANLINE,exist_ok=True)

    face_files = glob.glob(REAL_FACE_DIR)[400:799]
    # face_files = glob.glob(REAL_FACE_DIR)[:10]

    for idx,img_path in enumerate(face_files):
        img = cv2.imread(img_path)
        h,w,_ = img.shape
        print(f"image shape:{img.shape}")
        spoof = img.copy()

        # Adding a scan line
        scanlines = np.zeros((h,w),dtype=np.float32)
        scanlines[::2,:] = 0.85
        scanlines[1::2,:] = 1.0
        spoof = np.clip(spoof*scanlines[:,:,None],0,255).astype(np.uint8)

        ## diagonal screen glare reflection
        glare = np.zeros((h,w),dtype=np.uint8)
        cv2.line(glare,(0,int(h*0.2)),(w,int(h*0.8)),255,thickness=int(w*0.15))
        glare = cv2.GaussianBlur(glare,(71,71),0)
        spoof = cv2.addWeighted(spoof,1.0,cv2.cvtColor(glare,cv2.COLOR_GRAY2BGR),0.35,0)
        print(f"Spoof data val:{spoof}")
        print(f"ID value :{idx}")
        cv2.imwrite(f"{OUTPUT_SCANLINE}/synthetic_spoof_{idx:04d}.jpg",spoof)
        p = Path(f"{LABEL_PATH}/synthetic_spoof_{idx:04d}.txt")
        p.touch()

def convert_to_cutout_print():
    os.makedirs(OUTPUT_CUTOUT,exist_ok=True)

    face_files = glob.glob(REAL_FACE_DIR)[800:1199]
    for idx,img_path in enumerate(face_files):
        img = cv2.imread(img_path)
        bordered = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=[245,245,245])
        print(f"bordered data : {bordered}")

        h,w = bordered.shape[:2]
        src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst_pts = np.float32([[15,10],[w-20,5],[w-5,h-15],[20,h-5]])
        matrix = cv2.getPerspectiveTransform(src_pts,dst_pts)
        warped = cv2.warpPerspective(bordered,matrix,(w,h))
        cv2.imwrite(f"{OUTPUT_CUTOUT}/synthetic_cutout_{idx:04d}.jpg",warped)
        p = Path(f"{LABEL_PATH}/synthetic_cutout_{idx:04d}.txt")
        p.touch()
        print("Execution complete !")

img_to_sketch()
convert_to_screen_glare()
convert_to_cutout_print()




