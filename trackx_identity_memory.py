
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.utils.reid import ReID


# ============== TRACKX SETTINGS ==============

MATCH_THRESHOLD = 0.70
MATCH_MARGIN = 0.04
MAX_SAMPLES = 15
MIN_PERSON_HEIGHT = 100
SAMPLE_INTERVAL = 15

# ==============================================


class TrackXIdentityMemory:

    def __init__(self):
        self.gallery = {}
        self.track_map = {}
        self.next_id = 1
        self.frame_count = 0

    def normalize(self, embedding):
        feature = np.asarray(
            embedding, dtype=np.float32
        ).reshape(-1)

        norm = np.linalg.norm(feature)

        if norm < 1e-12:
            return None

        return feature / norm

    def compare(self, feature, person_id):
        samples = self.gallery[person_id]

        scores = [
            float(np.dot(feature, sample))
            for sample in samples
        ]

        # Average the three strongest sample matches.
        scores.sort(reverse=True)
        top_scores = scores[:3]

        return float(np.mean(top_scores))

    def add_sample(self, person_id, feature):
        samples = self.gallery[person_id]

        if len(samples) >= MAX_SAMPLES:
            return

        samples.append(feature.copy())

    def create_person(self, track_id, feature):
        person_id = self.next_id
        self.next_id += 1

        self.gallery[person_id] = [feature.copy()]
        self.track_map[track_id] = person_id

        print(f"NEW ID: PERSON {person_id:03d}")

        return person_id

    def identify_frame(self, track_ids, embeddings, heights):

        self.frame_count += 1

        assignments = [None] * len(track_ids)
        used_ids = set()

        features = [
            self.normalize(e) if e is not None else None
            for e in embeddings
        ]

        # PASS 1: Preserve identities of existing active tracks.
        for i, track_id in enumerate(track_ids):

            feature = features[i]

            if feature is None or heights[i] < MIN_PERSON_HEIGHT:
                continue

            if track_id in self.track_map:

                person_id = self.track_map[track_id]

                assignments[i] = person_id
                used_ids.add(person_id)

                # Add occasional samples from established tracks.
                if self.frame_count % SAMPLE_INTERVAL == 0:
                    self.add_sample(person_id, feature)

        # PASS 2: Match new tracks against the session gallery.
        for i, track_id in enumerate(track_ids):

            if assignments[i] is not None:
                continue

            feature = features[i]

            if feature is None or heights[i] < MIN_PERSON_HEIGHT:
                continue

            candidates = []

            for person_id in self.gallery:

                if person_id in used_ids:
                    continue

                score = self.compare(feature, person_id)
                candidates.append((score, person_id))

            candidates.sort(reverse=True)

            if candidates:

                best_score, best_id = candidates[0]

                second_score = (
                    candidates[1][0]
                    if len(candidates) > 1
                    else -1.0
                )

                print(
                    f"TRACK {track_id} | "
                    f"BEST MATCH PERSON {best_id:03d} | "
                    f"SCORE {best_score:.4f}"
                )

                if (
                    best_score >= MATCH_THRESHOLD
                    and best_score - second_score >= MATCH_MARGIN
                ):

                    person_id = best_id

                    self.track_map[track_id] = person_id
                    assignments[i] = person_id
                    used_ids.add(person_id)

                    print(
                        f"IDENTITY RECOVERED: "
                        f"PERSON {person_id:03d}"
                    )

                    self.add_sample(person_id, feature)
                    continue

            # No sufficiently reliable match.
            person_id = self.create_person(track_id, feature)

            assignments[i] = person_id
            used_ids.add(person_id)

        return assignments


def get_color(person_id):
    return (
        (person_id * 67) % 255,
        (person_id * 137) % 255,
        (person_id * 197) % 255
    )


# ============== START TRACKX ==============

print("Loading YOLO26 detector...")

detector = YOLO("yolo26n.pt")

print("Loading ReID encoder...")

encoder = ReID(
    "yolo26n-reid.onnx",
    device="cpu"
)

memory = TrackXIdentityMemory()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not open.")
    exit()

print("\n================================")
print(" TRACKX IDENTITY MEMORY STARTED")
print(" Session-only memory enabled")
print(" Press Q to exit")
print("================================\n")


while True:

    success, frame = camera.read()

    if not success:
        print("Camera frame unavailable.")
        break

    results = detector.track(
        frame,
        persist=True,
        tracker="trackx_botsort.yaml",
        classes=[0],
        conf=0.4,
        device="cpu",
        verbose=False
    )

    display = frame.copy()
    boxes = results[0].boxes

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

        embeddings = encoder(frame, person_boxes)

        person_ids = memory.identify_frame(
            track_ids,
            embeddings,
            heights
        )

        for i, person_id in enumerate(person_ids):

            if person_id is None:
                continue

            x1, y1, x2, y2 = coordinates[i].astype(int)

            color = get_color(person_id)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"PERSON {person_id:03d}"

            cv2.putText(
                display,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # Session statistics.
    cv2.putText(
        display,
        f"UNIQUE PEOPLE: {len(memory.gallery)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "TRACKX - SESSION IDENTITY MEMORY",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("\nTRACKX SESSION CLOSED")
print("Total unique people:", len(memory.gallery))
print("All session identities cleared.")
