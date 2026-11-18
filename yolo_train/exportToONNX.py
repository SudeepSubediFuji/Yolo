from ultralytics import YOLO



model = YOLO("runs/detect/train/weights/best.pt")
export_path = model.export(format="onnx", dynamic=True, simplify=True)

image = "./img.jpeg"
pt = YOLO("./best.pt")(image,rect=False,verbose=True)[0]
onnx = YOLO(export_path)(image,verbose=True)[0]

print(f"pt: {len(pt.boxes)} pills , mean conf {pt.boxes.conf.mean():.3f}")
print(f"onnx: {len(onnx.boxes)} pills , mean conf {onnx.boxes.conf.mean():.3f}")
print(f"max box difference: {(pt.boxes.xyxy - onnx.boxes.xyxy).abs.max():.3f}")

## validation comparoson