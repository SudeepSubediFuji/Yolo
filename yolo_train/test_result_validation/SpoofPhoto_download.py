import os
from datasets import load_dataset

# Open dataset with printouts, 2D screens, and mask attacks
dataset = load_dataset(
    "Ethanj7750/face-anti-spoofing-dataset", split="train", streaming=True
)

OUTPUT_DIR = "./unprocessed_spoofs/masks_and_screens"
os.makedirs(OUTPUT_DIR, exist_ok=True)

saved_count = 0
for item in dataset:
  # Filter for attacks only (paper prints, screen replay, 3D masks)
  if item.get("label") != "genuine":
    img = item["image"]
    img.save(f"{OUTPUT_DIR}/spoof_attack_{saved_count:05d}.jpg")
    saved_count += 1
    if saved_count >= 400:
      break

print(f"Downloaded {saved_count} spoof attack images.")