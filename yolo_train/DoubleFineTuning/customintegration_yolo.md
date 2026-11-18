To integrate your 3-class model (`0: real_face`, `1: intruder_face_mask`, `2: intruder_cardboard_mask`) into your existing `model.track()` pipeline, you have two main architectural choices depending on whether you want your attendance/security system to **silently ignore intruders** or **actively trigger an alert** when an intruder appears.

### 1. How `model.track()` Handles Multiple Classes

By default, Ultralytics' `model.track()` tracks **all classes** (`0`, `1`, and `2`) simultaneously and attaches both a `track_id` and a `cls` (class ID) to every tracked bounding box.

* You can read the class ID for every tracked face via `box.cls[0]` (returns `0.0`, `1.0`, or `2.0`).
* You can also filter classes directly inside `model.track(..., classes=[0])` if you only want the tracker to track legitimate faces.

---

### 2. Recommended Production Implementation (Track All + Alert on Intruders)

For a real-world attendance or security system, **track all three classes (`[0, 1, 2]`)** so that an intruder wearing a mask or holding cardboard still gets a persistent `track_id` across frames—allowing you to log a security event or trigger an alarm without granting attendance:

```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/stage2_clean_unfrozen/weights/best.pt")

# Class mapping matching your data.yaml
CLASS_NAMES = {
    0: "real_face",
    1: "intruder_face_mask",
    2: "intruder_cardboard_mask",
}

cap = cv2.VideoCapture(0)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  # Track all 3 classes so intruders also keep persistent track IDs
  results = model.track(
      source=frame,
      persist=True,  # Crucial: keeps track IDs consistent across frames
      conf=0.40,
      iou=0.50,
      classes=[0, 1, 2],  # Track real faces AND intruders
      verbose=False,
  )

  boxes = results[0].boxes

  if boxes is not None and len(boxes) > 0:
    for box in boxes:
      cls_id = int(box.cls[0].item())
      conf = float(box.conf[0].item())
      x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

      # Safe track ID extraction (in case tracker hasn't assigned ID on frame 1)
      track_id = int(box.id[0].item()) if box.id is not None else -1

      if cls_id == 0:
        # --- LEGITIMATE USER (real_face) ---
        label = f"ID:{track_id} Real ({conf:.2f})"
        color = (0, 255, 0)  # Green
        # -> Run your existing face recognition / attendance logging here

      elif cls_id in (1, 2):
        # --- SPOOF / INTRUDER DETECTED ---
        spoof_type = CLASS_NAMES[cls_id]
        label = f"ALERT ID:{track_id} {spoof_type} ({conf:.2f})"
        color = (0, 0, 255)  # Red
        # -> Block attendance & trigger security alert / snapshot

      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cv2.putText(
          frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
      )

  cv2.imshow("Face Detection & Anti-Spoof Tracking", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()

```

---

### 3. Two Crucial Edge-Case Safeguards for Tracking Spoofs

1. **Prevent Track ID Switching (`real_face` $\leftrightarrow$ `spoof_face`):**
* In Ultralytics trackers (`BoT-SORT` / `ByteTrack`), trackers associate boxes across frames primarily by **IoU overlap and class ID**. If a borderline cardboard mask flickers between class `0` and class `2` for a single frame, the tracker may spawn a new `track_id`.
* **Security Rule:** Maintain a small dictionary `intruder_ids = set()` in your Python loop. Once a `track_id` is flagged as class `1` or `2` (or overlaps heavily with a class `1`/`2` box), add its ID to `intruder_ids` so it stays permanently blocked for that session even if a single frame glitches back to class `0`.


2. **If You Only Want to Ignore Intruders Completely:**
* If your existing system simply needs to ignore classes `1` and `2` without changing any downstream logic, pass **`classes=[0]`** directly into `model.track(source=frame, persist=True, classes=[0])`. Ultralytics will automatically filter out classes `1` and `2` before returning `results[0].boxes`.