# Hard Negative Mining (HNM) Breakdown & Strategy

## 1. What is actually happening?
When you train a standard YOLO model, it learns positive features (what a face looks like). However, it rarely learns **negative constraints** (what is *not* a face). 
By feeding the model an image of a sketch and providing an **empty text file**, the loss function calculates a massive penalty for any bounding box the model tries to draw. You are mathematically forcing the network to learn the difference between human skin texture and graphite/pixels.

## 2. Crucial Pitfalls (What to watch out for)

### A. The "Ratio of Death" (Class Imbalance)
If your dataset has 1,000 real faces, and you suddenly add 5,000 empty background/sketch images, your model will experience "feature collapse." It will become terrified of making a mistake and stop detecting real faces entirely.
* **The Fix:** Hard negatives should make up **no more than 5% to 10%** of your total training data. If you have 5,000 real face images, add a maximum of 500 hard negative images.

### B. The Confidence Sweet Spot
Do not mine false positives that the model is already uncertain about. If the model detects a sketch with `conf=0.05`, it's basically guessing. 
* **The Fix:** Only mine images where the model is confident it found a face (e.g., `conf > 0.25` or `0.30`). These are the "hard" negatives that actually need correcting.

### C. Data Leakage
**Never** mine hard negatives from your validation set. You must source these from external folders (e.g., downloading a separate dataset of sketches, anime faces, and empty office rooms).

## 3. How to Evaluate and Compare the Results

After retraining your model with the newly added hard negatives, you need to compare `best.pt` (before HNM) and `best.pt` (after HNM).

### Look at `results.csv`:
1. **Precision (`metrics/precision(B)`):** This **MUST** go up. Higher precision means fewer false positives. This is the primary proof your HNM worked.
2. **Recall (`metrics/recall(B)`):** This will likely drop slightly. That is normal. However, if recall drops by a massive amount (e.g., from 0.85 to 0.60), you added too many hard negatives, and the model is now under-predicting.
3. **Classification Loss (`val/cls_loss`):** This should decrease on the validation set, indicating the model is better at distinguishing the `person` class from the background.

### The Real-World "Eye Test":
Run both models side-by-side on a live webcam feed while holding up a printed photo or sketch of a face to the camera. The pre-HNM model will draw a box; the post-HNM model should ignore it.


### So in Summary 
we are giving the penalty by using the labels. As if we created an empty label attach for picture that doesn't consist any face (as we are using the bbox and coco dataset value of face and excluding face photo data there). And then we train the model to use those labels and recreate a model that has higher confidence as it have already gotten hard negative value for the non-face object(such as the sketch ,and other human face like object.).