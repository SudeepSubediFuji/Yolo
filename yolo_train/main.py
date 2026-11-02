from ultralytics import YOLO

save_dir= "./trained_model"
model = YOLO("yolo26n.pt")

train_model = model.train(data="data.yaml",epochs=50,workers=2,device=0,imgsz=640,name="human_training",save_dir=save_dir,verbose=True)
## batch=-1 -> GPU memory's 60% usage
## batch = 0.7 -> 70% of Gpu memory
## train_model = model.train(data="data.yaml",epochs=500,batch=-1,workers=4,device=0,imgsz=640,name="human_training",save_dir=save_dir,verbose=True)