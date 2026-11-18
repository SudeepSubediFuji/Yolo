from ultralytics import YOLO
import cv2
import os

project="./training1_complete"
name="run_stage1"
model = "./runs/detect/DoubleFineTuning2/run_stage2_part1/weights/best.pt"

base_name = "messi"
test_img_folder = "./check"
test_img = f"{test_img_folder}/images/{base_name}.jpeg"
labels_dir = os.path.join(test_img_folder, "labels")
os.makedirs(labels_dir,exist_ok=True)
output_dir = "./check"
label_lines = []
img = cv2.imread(test_img)
h, w, _ = img.shape
label_path = os.path.join(labels_dir, f"{base_name}.txt")

load_model = YOLO(model)
result = load_model.predict(test_img,conf=0.5,iou=0.4,project="./model_validate",name="validate1",verbose=True)[0]

for id,box in enumerate(result.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    class_id = int(box.cls[0])
    x_c,y_c,w_c,h_c = box.xywhn[0].tolist()
    if int(box.cls[0]) == 0:
        print(f"Human detected at { box.xyxy[0].tolist()}")
        box_color_b = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color_b, 2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")
        cv2.imwrite(f'{output_dir}/{base_name}_{id}.png', img)
        label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
            print("label created successfully!")
    elif int(box.cls[0]) == 1:
        print(f"{box.xyxy[0].tolist()} になりしますを検知しました。")
        box_color_g=(0,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),box_color_g,2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")  
        cv2.imwrite(f"{output_dir}/{base_name}_{id}.png",img)  
        label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
            print("label created successfully!")
    else:
        print(f"{box.xyxy[0].tolist()} にカードボードを検知しました。")
        box_color_r=(0,0,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),box_color_r,2)
        print(f"Normalized value of output{id}.png for labelling:{class_id} {x_c:.6f} {y_c:.6f} {w_c:.6f} {h_c:.6f}")  
        cv2.imwrite(f"{output_dir}/{base_name}_{id}.png",img)
        label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
            print("label created successfully!")  