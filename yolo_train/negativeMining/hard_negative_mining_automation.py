import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- Configuration ---
SPOOF_DATA_DIR = "./unprocessed_spoofs"  # Folder full of sketches, photos, empty rooms
TRAIN_IMAGES_DIR = "./dataset/images/train"
TRAIN_LABELS_DIR = "./dataset/labels/train"
MODEL_PATH = "runs/detect/human_training/weights/best.pt"

# The threshold for what constitutes a "Hard" negative.
# If the model is > 25% sure a sketch is a face, we must penalize it.
CONFIDENCE_THRESHOLD = 0.25  
MAX_NEGATIVES_TO_ADD = 300  # Enforce the 5-10% ratio rule

def mine_hard_negatives():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    spoof_folder = Path(SPOOF_DATA_DIR)
    if not spoof_folder.exists():
        print(f"Error: {SPOOF_DATA_DIR} not found.")
        return

    valid_extensions = {".jpg", ".jpeg", ".png"}
    images_to_test = [p for p in spoof_folder.iterdir() if p.suffix.lower() in valid_extensions]
    
    print(f"Found {len(images_to_test)} candidate images. Starting mining...")
    
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    
    mined_count = 0
    
    for img_path in images_to_test:
        if mined_count >= MAX_NEGATIVES_TO_ADD:
            print(f"\nReached max hard negative limit ({MAX_NEGATIVES_TO_ADD}). Stopping.")
            break
            
        # Run inference
        results = model.predict(source=str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        
        # If the model found a bounding box in an image we KNOW is fake/empty
        if len(boxes) > 0:
            print(f"[MINED] False Positive found in: {img_path.name} (Conf: {boxes.conf[0]:.2f})")
            
            # 1. Copy image to training set
            dest_img_path = os.path.join(TRAIN_IMAGES_DIR, img_path.name)
            shutil.copy(img_path, dest_img_path)
            
            # 2. Create the EMPTY label file
            # e.g., sketch_01.jpg -> sketch_01.txt
            label_name = img_path.stem + ".txt"
            dest_label_path = os.path.join(TRAIN_LABELS_DIR, label_name)
            
            with open(dest_label_path, 'w') as f:
                pass  # Writing absolutely nothing
                
            mined_count += 1

    print("-" * 40)
    print(f"Mining Complete! Successfully added {mined_count} hard negatives to your dataset.")
    print("Next Step: Re-run your YOLO training script to fine-tune on these new constraints.")

if __name__ == "__main__":
    mine_hard_negatives()