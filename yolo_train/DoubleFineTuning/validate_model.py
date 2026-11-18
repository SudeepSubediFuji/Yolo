from ultralytics import YOLO
import cv2
import os

project="./training1_complete"
name="run_stage1"

# -2
model = "./runs/detect/DoubleFineTuning2/run_stage2_part1/weights/best.pt"
# "/home/user/yolo_train/hnm/runs/detect/DoubleFineTuning/run_stage2_part3-3/weights/best.pt"

yolo_model = YOLO("yolo26n.pt")

# REAL_FACE_IMG_DIR = "./training_data/images/train/realfaceSulpture.jpg"
## 20 is remaining
# 2026-11-12-134624
# unknown_20261027_132940

## 7 ,8
## 
base_name = "SELF_TEST"
test_img = f"./test_img/{base_name}.jpg"
# test_img = f"./test_img/image/{base_name}.jpeg"
test_img_folder = "./test_img"
# labels_dir = os.path.join(test_img_folder, "labels")
output_dir = "./output_img"
label_lines = []
img = cv2.imread(test_img)
h, w, _ = img.shape

load_model = YOLO(model)

result = load_model.predict(test_img,conf=0.5,project="./model_validate",name="validate1",verbose=True)[0]

for id,box in enumerate(result.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    class_id = int(box.cls[0])
    x_c,y_c,w_c,h_c = box.xywhn[0].tolist()
    if int(box.cls[0]) == 0:
        print(f"Human detected at { box.xyxy[0].tolist()}")
        # label_path = os.path.join(labels_dir, f"{base_name}.txt")
        box_color_b = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color_b, 2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")
        # label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        cv2.imwrite(f'{output_dir}/{base_name}_{id}.png', img)
    # Save the .txt label file (even if empty, which tells YOLO there are no objects)
        # with open(label_path, "w") as f:
        #     f.write("\n".join(label_lines))
        #     print("label created successfully!")
    elif int(box.cls[0]) == 1:
        print(f"{box.xyxy[0].tolist()} になりしますを検知しました。")
        box_color_g=(0,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),box_color_g,2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")  
        cv2.imwrite(f"{output_dir}/{base_name}_{id}.png",img)  
    else:
        print(f"{box.xyxy[0].tolist()} にカードボードを検知しました。")
        box_color_r=(0,0,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),box_color_r,2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")  
        cv2.imwrite(f"{output_dir}/{base_name}_{id}.png",img)  

        
        


## not match 
## cardboardface_04,cardboardface_13

# Normalized value of output1.png for labelling:0 0.571113 0.437901 0.143408 0.213329
# Human detected at [142.89480590820312, 158.177978515625, 295.3341979980469, 367.52801513671875]
# Normalized value of output2.png for labelling:0 0.273893 0.328566 0.190549 0.261688

## 0.541695 0.586463 0.282603 0.489407
##  0 0.167745 0.475844 0.333558 0.951501
## 2 0.487206 0.507441 0.230121 0.431194