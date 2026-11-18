from ultralytics import YOLO
import cv2
import os

# Directories
test_img_folder = "./test_img"
test_img = "./test_img/image/"
labels_dir = os.path.join(test_img_folder, "labels")
output_img_dir = "./output_img"

# Create required folders if they don't exist
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Load model (using standard YOLO model to auto-label people as class 0)
yolo_model = YOLO("yolo26n.pt")

# If you want to use your custom trained model later, uncomment below:
# custom_model_path = "./runs/detect/finaltrainig2_complete/run_stage2/weights/best.pt"
# yolo_model = YOLO(custom_model_path)

image_extension = (".jpeg", ".jpg", ".png", ".webp", ".heic", ".heif")
files_path = [
    f for f in os.listdir(test_img)
    if f.lower().endswith(image_extension)
]
print(f"Found {len(files_path)} images: {files_path}")

for file in files_path:
    img_path = os.path.join(test_img, file)
    base_name = os.path.splitext(file)[0]
    label_path = os.path.join(labels_dir, f"{base_name}.txt")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {file}")
        continue

    # Run prediction per image so bounding boxes match the specific image
    result = yolo_model.predict(img_path, conf=0.4, verbose=False)[0]
    
    label_lines = []
    h_img, w_img, _ = img.shape

    for box in result.boxes:
        class_id = int(box.cls[0])
        
        # Class 0 in standard YOLO is 'person' (matches your 0: real_person)
        if class_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
            
            # Draw bounding box on preview image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Get normalized YOLO format: class_id x_center y_center width height
            x_c, y_c, w, h = box.xywhn[0].tolist()
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Save the .txt label file (even if empty, which tells YOLO there are no objects)
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))
        
    # Save visual preview image with matching filename
    cv2.imwrite(os.path.join(output_img_dir, f"{base_name}.png"), img)
    print(f"Saved {len(label_lines)} label(s) to {label_path}")
