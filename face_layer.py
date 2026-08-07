import os
import cv2
import numpy as np
import face_recognition
from db import NovaDB

FACE_MATCH_THRESHOLD = 0.5
ENCODING_JITTER_REALTIME = 1
ENCODING_JITTER_ENROLL = 5
VIDEO_FRAME_SAMPLE_RATE = 10
MIN_FACE_SIZE = 40

def encode_face_from_image(image_path: str,
                           jitter: int = ENCODING_JITTER_ENROLL
                           ) -> list[np.ndarray]:
    img = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(img, model="hog")

    if not locations:
        return []

    locations = [
        loc for loc in locations
        if (loc[2] - loc[0]) >= MIN_FACE_SIZE
    ]

    if not locations:
        return []

    encodings = face_recognition.face_encodings(img, locations,
                                                num_jitters=jitter)
    return encodings

def encode_face_from_frame(frame_bgr: np.ndarray,
                           jitter: int = ENCODING_JITTER_REALTIME
                           ) -> list[tuple[tuple, np.ndarray]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

    locations = face_recognition.face_locations(small, model="hog")
    locations = [
        loc for loc in locations
        if (loc[2] - loc[0]) >= MIN_FACE_SIZE // 2
    ]

    if not locations:
        return []

    encodings = face_recognition.face_encodings(small, locations,
                                                num_jitters=jitter)

    scaled = []
    for (top, right, bottom, left), enc in zip(locations, encodings):
        scaled.append((
            (top * 2, right * 2, bottom * 2, left * 2),
            enc
        ))

    return scaled

def encode_faces_from_video(video_path: str,
                            jitter: int = ENCODING_JITTER_ENROLL
                            ) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    encodings = []

    print(f"  [VIDEO] {os.path.basename(video_path)} "
          f"— {total} frames, sampling every {VIDEO_FRAME_SAMPLE_RATE}th")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % VIDEO_FRAME_SAMPLE_RATE != 0:
            continue

        found = encode_face_from_frame(frame, jitter=1)
        for _, enc in found:
            encodings.append(enc)

    cap.release()
    print(f"  [VIDEO] Extracted {len(encodings)} face encoding(s).")
    return encodings

def enroll_person_faces(person_id: int, db: NovaDB,
                        data_dir: str = "known_faces") -> int:
    person = db.get_person(person_id)
    if not person:
        print(f"[ERROR] No person with id={person_id}")
        return 0

    folder_name = person["first_name"].lower()
    person_dir = os.path.join(data_dir, folder_name)

    if not os.path.exists(person_dir):
        for d in os.listdir(data_dir):
            if d.lower() == folder_name:
                person_dir = os.path.join(data_dir, d)
                break
        else:
            print(f"[ERROR] Folder not found: {person_dir}")
            return 0

    files = os.listdir(person_dir)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv"}

    image_files = [f for f in files
                   if os.path.splitext(f)[1].lower() in img_exts]
    video_files = [f for f in files
                   if os.path.splitext(f)[1].lower() in vid_exts]

    print(f"\n[ENROLL FACE] {person['first_name']} {person['last_name']}")
    print(f"  {len(image_files)} image(s), {len(video_files)} video(s)")

    all_encodings = []

    for img_name in image_files:
        img_path = os.path.join(person_dir, img_name)
        found = encode_face_from_image(img_path,
                                       jitter=ENCODING_JITTER_ENROLL)
        if not found:
            print(f"  [WARN] No face in {img_name}")
            continue
        all_encodings.extend(found)
        print(f"  [OK] {img_name} → {len(found)} encoding(s)")

    for vid_name in video_files:
        vid_path = os.path.join(person_dir, vid_name)
        found = encode_faces_from_video(vid_path,
                                        jitter=ENCODING_JITTER_ENROLL)
        all_encodings.extend(found)

    if not all_encodings:
        print(f"  [ERROR] No usable face data found for "
              f"{person['first_name']}.")
        return 0

    db.save_face_encodings(person_id, all_encodings)
    print(f"  [OK] {len(all_encodings)} total encoding(s) saved to DB.")
    return len(all_encodings)

def enroll_person_faces_from_folder(person_id: int, db: NovaDB,
                                    folder: str,
                                    append: bool = False) -> int:
    person = db.get_person(person_id)
    if not person:
        print(f"[ERROR] No person with id={person_id}")
        return 0
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return 0

    files = os.listdir(folder)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv"}

    image_files = sorted(f for f in files
                         if os.path.splitext(f)[1].lower() in img_exts)
    video_files = sorted(f for f in files
                         if os.path.splitext(f)[1].lower() in vid_exts)

    print(f"\n[ENROLL FACE — PATH] {person['first_name']} {person['last_name']}")
    print(f"  {len(image_files)} image(s), {len(video_files)} video(s) in {folder}")

    all_encodings = []
    for img_name in image_files:
        img_path = os.path.join(folder, img_name)
        found = encode_face_from_image(img_path, jitter=ENCODING_JITTER_ENROLL)
        if not found:
            print(f"  [WARN] No face in {img_name}")
            continue
        all_encodings.extend(found)
        print(f"  [OK] {img_name} → {len(found)} encoding(s)")

    for vid_name in video_files:
        vid_path = os.path.join(folder, vid_name)
        found = encode_faces_from_video(vid_path, jitter=ENCODING_JITTER_ENROLL)
        all_encodings.extend(found)

    if not all_encodings:
        print(f"  [ERROR] No usable face data found in {folder}.")
        return 0

    if append:
        total = db.add_face_encodings(person_id, all_encodings)
    else:
        db.save_face_encodings(person_id, all_encodings)
        total = len(all_encodings)

    print(f"  [OK] {total} total encoding(s) saved to DB.")
    return total

class FaceLayer:
    def __init__(self, db: NovaDB,
                 threshold: float = FACE_MATCH_THRESHOLD):
        self.db = db
        self.threshold = threshold
        self._known: list[tuple[int, str, np.ndarray]] = []
        self.reload()

    def reload(self):
        self._known = self.db.load_all_face_encodings()
        print(f"[FaceLayer] Loaded {len(self._known)} encoding(s) "
              f"for {len({p for p, _, _ in self._known})} person(s).")

    AMBIGUITY_MARGIN = 0.08

    def identify(self, frame_bgr: np.ndarray
                 ) -> list[dict]:
        if not self._known:
            return []

        found = encode_face_from_frame(frame_bgr)
        if not found:
            return []

        known_encodings = [enc for _, _, enc in self._known]
        results = []

        for location, encoding in found:
            distances = face_recognition.face_distance(
                known_encodings, encoding
            )
            order = np.argsort(distances)
            best_idx = int(order[0])
            best_dist = float(distances[best_idx])
            best_pid = self._known[best_idx][0]

            runner_up_name = None
            runner_up_dist = None
            for idx in order[1:]:
                idx = int(idx)
                candidate_pid = self._known[idx][0]
                if candidate_pid != best_pid:
                    runner_up_name = self._known[idx][1]
                    runner_up_dist = float(distances[idx])
                    break

            if best_dist <= self.threshold:
                pid, name, _ = self._known[best_idx]
                confidence = round(1.0 - (best_dist / self.threshold), 3)
                confidence = max(0.0, min(1.0, confidence))
                is_known = True
            else:
                pid = None
                name = "Unknown"
                confidence = 0.0
                is_known = False

            runner_up_conf = None
            ambiguous = False
            if runner_up_dist is not None:
                runner_up_conf = round(
                    max(0.0, min(1.0, 1.0 - (runner_up_dist / self.threshold))), 3)
                if is_known and (runner_up_dist - best_dist) < self.AMBIGUITY_MARGIN:
                    ambiguous = True

            results.append({
                "person_id": pid,
                "name": name,
                "confidence": confidence,
                "distance": best_dist,
                "location": location,
                "is_known": is_known,
                "runner_up_name": runner_up_name,
                "runner_up_conf": runner_up_conf,
                "ambiguous": ambiguous,
            })

        return results

    def location_to_box(self, location: tuple) -> tuple:
        top, right, bottom, left = location
        return left, top, right - left, bottom - top

if __name__ == "__main__":
    import sys
    from db import NovaDB

    db = NovaDB()

    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        pid = int(sys.argv[2])
        enroll_person_faces(pid, db)

    else:
        print("[FaceLayer] Starting webcam test... press Q to quit.")
        layer = FaceLayer(db)

        if not layer._known:
            print("[WARN] No face encodings in database.")
            print("       Run: python3 face_layer.py enroll <person_id>")
            sys.exit(0)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = layer.identify(frame)

            for r in results:
                x, y, w, h = layer.location_to_box(r["location"])
                color = (0, 220, 0) if r["is_known"] else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = (f"{r['name']} {r['confidence']:.0%}"
                         if r["is_known"] else "Unknown")
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Nova — Face Layer Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
