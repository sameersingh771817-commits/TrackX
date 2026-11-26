
import cv2
import time
from collections import deque
from ultralytics import YOLO
from ultralytics.trackers.utils.reid import ReID

from trackx_memory import TrackXIdentityMemory, get_color


class TrackXEngine:

    def __init__(self):
        print("Loading TrackX Performance Engine...")

        self.detector = YOLO("yolo26n.pt")
        self.encoder = ReID("yolo26n-reid.onnx", device="cpu")

        self.memory = TrackXIdentityMemory()

        self.frame_count = 0
        self.reid_interval = 5

        self.total_times = deque(maxlen=30)
        self.yolo_times = deque(maxlen=30)
        self.reid_times = deque(maxlen=30)

        self.last_processing_ms = 0
        self.last_fps = 0
        self.yolo_time_ms = 0
        self.reid_time_ms = 0

        print("TrackX Performance Engine Ready!")

    def process_frame(self, frame):

        total_start = time.perf_counter()

        self.frame_count += 1
        display = frame.copy()

        # YOLO detection and tracking
        yolo_start = time.perf_counter()

        results = self.detector.track(
            frame,
            persist=True,
            tracker="trackx_botsort.yaml",
            classes=[0],
            conf=0.4,
            imgsz=416,
            max_det=20,
            device="cpu",
            verbose=False
        )

        yolo_elapsed = (
            time.perf_counter() - yolo_start
        ) * 1000

        boxes = results[0].boxes
        active_ids = []

        # ReID and identity processing
        reid_start = time.perf_counter()

        if (
            boxes is not None
            and len(boxes) > 0
            and boxes.id is not None
        ):

            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().numpy()
            person_boxes = boxes.xywh.cpu().numpy()

            heights = [
                int(box[3] - box[1])
                for box in coordinates
            ]

            if self.frame_count % self.reid_interval == 1:

                embeddings = self.encoder(
                    frame,
                    person_boxes
                )

                person_ids = self.memory.identify_frame(
                    track_ids,
                    embeddings,
                    heights
                )

            else:

                person_ids = [
                    self.memory.track_map.get(track_id)
                    for track_id in track_ids
                ]

            for i, person_id in enumerate(person_ids):

                if person_id is None:
                    continue

                active_ids.append(person_id)

                x1, y1, x2, y2 = coordinates[i].astype(int)
                color = get_color(person_id)

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    display,
                    f"PERSON {person_id:03d}",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        reid_elapsed = (
            time.perf_counter() - reid_start
        ) * 1000

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        # Rolling performance averages
        self.total_times.append(total_elapsed)
        self.yolo_times.append(yolo_elapsed)
        self.reid_times.append(reid_elapsed)

        self.last_processing_ms = (
            sum(self.total_times) / len(self.total_times)
        )

        self.yolo_time_ms = (
            sum(self.yolo_times) / len(self.yolo_times)
        )

        self.reid_time_ms = (
            sum(self.reid_times) / len(self.reid_times)
        )

        self.last_fps = (
            1000 / self.last_processing_ms
            if self.last_processing_ms > 0
            else 0
        )

        unique_people = len(self.memory.gallery)
        active_people = len(set(active_ids))

        return display, unique_people, active_people
