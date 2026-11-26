
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo26n.pt")

print("YOLO MODEL LOADED SUCCESSFULLY!")
print("TrackX is ready for object detection!")
