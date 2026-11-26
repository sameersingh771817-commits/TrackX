
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

tracks = {}
next_id = 1
MAX_MISSED = 15
IOU_THRESHOLD = 0.3


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, conf=0.4, verbose=False)[0]

    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            detections.append({
                "box": box.tolist(),
                "class": int(cls),
                "confidence": float(conf)
            })

    # Find possible matches between old tracks and new detections
    candidates = []

    for track_id, track in tracks.items():
        for index, detection in enumerate(detections):
            if track["class"] != detection["class"]:
                continue

            iou = calculate_iou(track["box"], detection["box"])

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, track_id, index))

    # Match highest IoU first
    candidates.sort(reverse=True)

    matched_tracks = set()
    matched_detections = set()

    for iou, track_id, index in candidates:
        if track_id not in matched_tracks and index not in matched_detections:
            tracks[track_id]["box"] = detections[index]["box"]
            tracks[track_id]["missed"] = 0

            matched_tracks.add(track_id)
            matched_detections.add(index)

    # Increase missed count for unmatched tracks
    for track_id in list(tracks.keys()):
        if track_id not in matched_tracks:
            tracks[track_id]["missed"] += 1

            if tracks[track_id]["missed"] > MAX_MISSED:
                del tracks[track_id]

    # Create new IDs for unmatched detections
    for index, detection in enumerate(detections):
        if index not in matched_detections:
            tracks[next_id] = {
                "box": detection["box"],
                "class": detection["class"],
                "missed": 0
            }
            next_id += 1

    # Draw tracking boxes and IDs
    for track_id, track in tracks.items():
        if track["missed"] != 0:
            continue

        x1, y1, x2, y2 = map(int, track["box"])

        class_name = model.names[track["class"]]
        label = f"ID {track_id}: {class_name}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("TrackX - Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
