from ultralytics import YOLO
import torch
import shutil
import gc
from pathlib import Path
import os

FALSE_DATA_DIR = "./false_data/train"
REAL_FACE_IMG_DIR = "./training_data/images/train"
REAL_FACE_LABELS_DIR = "./training_data/labels/train"

## Previously tinued with face-dataset with petter performance 
# (same data use for training might not be used for learning but rather remembering which we don't want)
# model_path = "./best.pt"

CONFIDENCE_THREASHOLD = 0.5
MAX_NEGATIVE = 214

def HardNegativeMining():
    print("** Started First phase(Hard Negative mining.)")
    model = YOLO("yolo26n.pt")

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

def DoubleFineTuning():
    print("** Sencond Phase Stage 1: Backbone freezing (Freeze=10)")
    model_s1 = YOLO("yolo26n.pt")

    model_s1.train(
        data="training_data.yaml",
        epochs=30,
        batch=36,
        imgsz=640,
        freeze=6,
        patience=7,
        lr0=0.001,
        lrf=0.0001,
        project="./DoubleFineTuning2",
        name="run_stage1_part1",
        device=0,
        verbose=True,
        hsv_v=0.4,
        hsv_s=0.7,
        degrees=15,
        scale=0.5,
        erasing=0.2,
        fliplr=0.5,
        perspective=0.0005,
        flipud=0.5,
        cls=1
        )
    stage1_best_model = "./runs/detect/DoubleFineTuning2/run_stage1_part1/weights/best.pt"
    
    if not os.path.exists(stage1_best_model):
        raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage1_best_model}")
    
    del model_s1
    gc.collect()
    torch.cuda.empty_cache()

    model_s2 = YOLO(stage1_best_model)

    ## deta augumentation parameter: hsv_v=0.4 hsv_s=0.7 degree=15 scale=0.5 eraser=0.2

    model_s2.train(
        data="training_data.yaml",
        epochs=30,
        batch=36,
        imgsz=640,
        freeze=0,
        patience=5,
        lr0=0.0001,
        lrf=0.01,
        device=0,
        verbose=True,
        project="./DoubleFineTuning2",
        name="run_stage2_part1",
        warmup_epochs=1.0,
        hsv_v=0,
        hsv_s=0,
        degrees=0,
        scale=0.5,
        erasing=0,
        cls=0.5,
        perspective=0.00
    )

    print(f"Stage 2 is completed and the checkpoint is stored in {"./runs/detect/DoubleFineTuning2/run_stage2_part1/weights/best.pt"}")

if __name__ == "__main__":
    # HardNegativeMining()
    DoubleFineTuning()

