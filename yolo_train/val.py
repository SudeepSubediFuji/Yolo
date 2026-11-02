from ultralytics import YOLO

trained_model_loc = "/home/user/yolo_train/trained_model/weights/best.pt"
trained_model = YOLO(trained_model_loc)
# model = YOLO("yolo26n.pt")

## Trained data validation
validate_model = trained_model.val(data="data.yaml",imgsz=640,batch=16,device=0)

print(validate_model.box.image_metrics)