
import numpy as np

MATCH_THRESHOLD = 0.70
MATCH_MARGIN = 0.04
MAX_SAMPLES = 15
MIN_PERSON_HEIGHT = 100
SAMPLE_INTERVAL = 15


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

        scores.sort(reverse=True)

        return float(np.mean(scores[:3]))

    def add_sample(self, person_id, feature):
        samples = self.gallery[person_id]

        if len(samples) < MAX_SAMPLES:
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

        for i, track_id in enumerate(track_ids):

            feature = features[i]

            if feature is None or heights[i] < MIN_PERSON_HEIGHT:
                continue

            if track_id in self.track_map:

                person_id = self.track_map[track_id]

                assignments[i] = person_id
                used_ids.add(person_id)

                if self.frame_count % SAMPLE_INTERVAL == 0:
                    self.add_sample(person_id, feature)

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
