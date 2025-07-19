import sys
sys.path.append('./yolo12')
from ultralytics import YOLO

model = YOLO('./yolo12/ckpts/yolov12l-seg.pt')
model.export(format="engine", imgsz=1280, dynamic=True, half=True)