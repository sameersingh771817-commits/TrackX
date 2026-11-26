
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo26n.pt")

print("TrackX Object Detection Started!")

# Start webcam detection
model.predict(
    source=0,
    show=True,
    conf=0.4
)
