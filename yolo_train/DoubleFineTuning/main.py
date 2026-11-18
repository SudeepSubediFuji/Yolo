from ultralytics import YOLO
from pathlib import Path
import os
import shutil

FALSE_DATA_DIR = "./false_data/train"
REAL_FACE_IMG_DIR = "./images/train"
REAL_FACE_LABELS_DIR = "./labels/train"

## Previously tinued with face-dataset with petter performance
model_path = "./best.pt"

CONFIDENCE_THREASHOLD = 0.5
MAX_NEGATIVE = 214

def hnm():
    print("Loading the model")
    model = YOLO(model_path)

    spoof_folder = Path(FALSE_DATA_DIR)
    img_extension = {".jpeg",".jpg",".png"}
    images_to_test = [p for p in spoof_folder.iterdir() if p.suffix.lower() in img_extension]

    print(f"Found image {len(images_to_test)} for mining")

    mined_count = 0
    for img_path in images_to_test:
        if mined_count >= MAX_NEGATIVE:
            print(f"Remaining to mine :{mined_count}")
            break
        results = model.predict(img_path,verbose=False,conf=CONFIDENCE_THREASHOLD)
        ## only selecting boxes with the 0/person
        boxes = results[0].boxes
        if len(boxes)>0:
            print(f"False positive found in : {img_path.name}, with confidence {boxes.conf[0]}")

            dest_img_path = os.path.join(REAL_FACE_IMG_DIR,img_path.name)
            shutil.copy(img_path,dest_img_path)

            label_name = img_path.stem + ".txt"
            dest_label_path = os.path.join(REAL_FACE_LABELS_DIR,label_name)

            with open(dest_label_path,"w"):
                pass

            mined_count += 1
    print("-" * 40)
    print(f"Total number of mined count: {mined_count}")
    print("** Hard Negative mining complete**")


hnm()
