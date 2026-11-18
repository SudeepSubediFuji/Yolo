from ultralytics import YOLO
import cv2
import os

project="./training1_complete"
name="run_stage1"

model = "./runs/detect/finaltrainig2_complete/run_stage2/weights/best.pt"
yolo_model = YOLO("yolo26n.pt")

# REAL_FACE_IMG_DIR = "./training_data/images/train/realfaceSulpture.jpg"
test_img_folder = "./test_img"
test_img = "./test_img/image/"

load_model = YOLO(model)

result = yolo_model.predict(test_img,conf=0.4,project="./model_validate",name="validate1",verbose=True)[0]

# print(files_path)
image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
files_path = [
    f
    for f in os.listdir(test_img)
    if f.lower().endswith(image_extension)
    ]
print(f"File path:{files_path}")

for id,file in enumerate(files_path):
    img_path = os.path.join(test_img, file)
    print(f"Image path: {img_path} and id:{id}")
    # 画像をファイルパスから実際に読み込みます
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img}")
        continue

    for box in result.boxes:
        if int(box.cls[0]) == 0:
            print(f"Human detected at { box.xyxy[0].tolist()}")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w, _ = img.shape
            box_color = (255, 0, 0)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            class_id = int(box.cls[0])
            x_c,y_c,w,h = box.xywhn[0].tolist()
            print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c,y_c,w,h}")
            cv2.imwrite(f'./output_img/output_{id}.png', img)