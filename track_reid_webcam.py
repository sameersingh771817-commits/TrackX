
from ultralytics import YOLO

# Load YOLO26 Nano
model = YOLO("yolo26n.pt")

print("TrackX Person Re-Identification Started")
print("Press Q in the webcam window to exit.")

# Track people using BoT-SORT with ReID
model.track(
    source=0,
    tracker="trackx_botsort.yaml",
    persist=True,
    show=True,
    conf=0.4,
    classes=[0],
    device="cpu"
)
