"""
clothing_layer.py

Clothing re-identification layer.

When face/voice confirms identity: extract clothing histograms from
torso, legs, and shoes zones and save to DB as wardrobe memory.

When face/voice is unavailable: compare current clothing against
wardrobe entries to attempt re-identification.

Handles partial visibility gracefully -- each zone is only sampled
if its pixel area is large enough to be meaningful.
"""

import cv2
import numpy as np
import time
from typing import Optional
from db import NovaDB

# ── Config ────────────────────────────────────────────────────────────────────

# HSV histogram bins
HUE_BINS = 18
SAT_BINS = 8
HIST_SIZE = HUE_BINS * SAT_BINS          # 144 per zone

# Zone minimum pixel area before sampling
MIN_ZONE_PIXELS = 800

# Clothing match threshold (cv2.HISTCMP_CORREL, 0-1, higher = better)
CLOTHING_MATCH_THRESHOLD = 0.65

# Torso gets more weight than legs (more visible, more distinctive)
TORSO_WEIGHT = 0.60
LEGS_WEIGHT  = 0.40

# How often to save a new wardrobe entry (seconds) per person
WARDROBE_LOG_INTERVAL = 10.0


# ── Zone extraction ───────────────────────────────────────────────────────────

def extract_zones(frame_bgr: np.ndarray,
                  face_box: tuple) -> dict:
    """
    Given a frame and the face bounding box (x, y, w, h),
    estimate body zones below the face and extract HSV histograms.

    Returns dict with keys: 'torso', 'legs', 'shoes'
    Each value is a numpy float32 array of shape (HIST_SIZE,)
    or None if the zone was too small to sample.

    Also returns 'visible_zones': list of zone names that were sampled.
    Also returns 'zone_rects': dict of (x,y,w,h) for each visible zone
    (useful for debug drawing).
    """
    fx, fy, fw, fh = face_box
    fh = max(fh, 1)
    fw = max(fw, 1)
    frame_h, frame_w = frame_bgr.shape[:2]

    # Body column: same horizontal span as face, slightly wider
    bx = max(0, fx - fw // 3)
    bw = min(fw + fw // 3 * 2, frame_w - bx)

    # Torso: chin to ~2.5x face height below
    torso_y = fy + fh
    torso_h = min(int(fh * 1.6), frame_h - torso_y)

    # Legs: below torso, another ~2x face height
    legs_y = torso_y + torso_h
    legs_h = min(int(fh * 1.8), frame_h - legs_y)

    # Shoes: bottom strip
    shoes_y = legs_y + legs_h
    shoes_h = min(int(fh * 0.8), frame_h - shoes_y)

    zones = {}
    zone_rects = {}
    visible_zones = []

    def sample(y, h, name):
        if h < 1 or y + h > frame_h:
            zones[name] = None
            return
        crop = frame_bgr[y:y+h, bx:bx+bw]
        if crop.size < MIN_ZONE_PIXELS:
            zones[name] = None
            return
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                             [HUE_BINS, SAT_BINS],
                             [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        zones[name] = hist.flatten().astype(np.float32)
        zone_rects[name] = (bx, y, bw, h)
        visible_zones.append(name)

    sample(torso_y, torso_h, "torso")
    sample(legs_y,  legs_h,  "legs")
    sample(shoes_y, shoes_h, "shoes")

    zones["visible_zones"] = visible_zones
    zones["zone_rects"]    = zone_rects
    return zones


def describe_zones(zones: dict) -> str:
    """
    Produce a human-readable clothing description from zone histograms.
    Maps dominant HSV hue to a colour name.
    """
    descriptions = []
    for zone_name in ("torso", "legs", "shoes"):
        hist = zones.get(zone_name)
        if hist is None:
            continue
        colour = _dominant_colour(hist)
        part   = {"torso": "top", "legs": "bottom", "shoes": "shoes"}[zone_name]
        descriptions.append(f"{colour} {part}")
    return ", ".join(descriptions) if descriptions else "not visible"


def _dominant_colour(hist: np.ndarray) -> str:
    """Map a flattened HSV histogram to a human colour name."""
    # Reshape to (HUE_BINS, SAT_BINS)
    h = hist.reshape(HUE_BINS, SAT_BINS)
    # Sum across saturation to get hue distribution
    hue_dist = h.sum(axis=1)
    low_sat   = h[:, :2].sum()   # low saturation = neutral colour
    total     = hue_dist.sum()

    if total == 0:
        return "unknown"

    # If most pixels are low saturation, it's black/white/gray
    if low_sat / total > 0.55:
        # Look at brightness in original -- approximated by hue distribution
        # Use hue bin 0 (near-zero) as proxy
        return "dark" if hue_dist[0] < hue_dist.mean() else "light"

    dominant_hue_bin = int(np.argmax(hue_dist))
    # Each bin = 180/18 = 10 degrees
    hue_degrees = dominant_hue_bin * 10

    if hue_degrees < 10 or hue_degrees >= 160:
        return "red"
    elif hue_degrees < 25:
        return "orange"
    elif hue_degrees < 35:
        return "yellow"
    elif hue_degrees < 85:
        return "green"
    elif hue_degrees < 130:
        return "blue"
    elif hue_degrees < 160:
        return "purple"
    return "unknown"


# ── Wardrobe matching ─────────────────────────────────────────────────────────

def match_wardrobe(db: NovaDB,
                   zones: dict) -> tuple[Optional[int], Optional[str], float]:
    """
    Compare current clothing zones against all wardrobe entries in DB.

    Returns (person_id, full_name, score) of best match,
    or (None, None, 0.0) if no match above threshold.
    """
    torso_hist = zones.get("torso")
    legs_hist  = zones.get("legs")

    if torso_hist is None and legs_hist is None:
        return None, None, 0.0

    all_entries = db.load_all_clothing_entries()
    if not all_entries:
        return None, None, 0.0

    best_pid   = None
    best_name  = None
    best_score = 0.0

    for pid, name, entry in all_entries:
        score = _compare_entry(torso_hist, legs_hist, entry)
        if score > best_score:
            best_score = score
            best_pid   = pid
            best_name  = name

    if best_score >= CLOTHING_MATCH_THRESHOLD:
        return best_pid, best_name, best_score
    return None, None, best_score


def _compare_entry(torso: Optional[np.ndarray],
                   legs:  Optional[np.ndarray],
                   entry: dict) -> float:
    """Weighted histogram correlation between current zones and one entry."""
    scores = []
    weights = []

    if torso is not None and entry.get("torso_hist") is not None:
        t = entry["torso_hist"].reshape(-1, 1).astype(np.float32)
        q = torso.reshape(-1, 1).astype(np.float32)
        scores.append(cv2.compareHist(q, t, cv2.HISTCMP_CORREL))
        weights.append(TORSO_WEIGHT)

    if legs is not None and entry.get("legs_hist") is not None:
        l = entry["legs_hist"].reshape(-1, 1).astype(np.float32)
        q = legs.reshape(-1, 1).astype(np.float32)
        scores.append(cv2.compareHist(q, l, cv2.HISTCMP_CORREL))
        weights.append(LEGS_WEIGHT)

    if not scores:
        return 0.0

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


# ── Clothing layer class ──────────────────────────────────────────────────────

class ClothingLayer:
    """
    Stateful clothing layer.

    Tracks last wardrobe save time per person to avoid
    flooding the DB with duplicate entries.
    """

    def __init__(self, db: NovaDB):
        self.db = db
        self._last_saved: dict[int, float] = {}

    def process(self, frame_bgr: np.ndarray,
                face_box: tuple) -> dict:
        """
        Extract clothing zones from frame + face_box.
        Returns zone dict (pass to save_to_wardrobe or match_wardrobe).
        """
        return extract_zones(frame_bgr, face_box)

    def save_to_wardrobe(self, person_id: int, zones: dict,
                          label: str = ""):
        """
        Save current clothing observation to DB wardrobe memory.
        Rate-limited per person to WARDROBE_LOG_INTERVAL seconds.
        """
        now  = time.time()
        last = self._last_saved.get(person_id, 0.0)
        if now - last < WARDROBE_LOG_INTERVAL:
            return

        torso = zones.get("torso")
        legs  = zones.get("legs")
        shoes = zones.get("shoes")

        if torso is None and legs is None:
            return  # nothing to save

        # Use zeros for absent zones so DB stays consistent
        torso = torso if torso is not None else np.zeros(HIST_SIZE, np.float32)
        legs  = legs  if legs  is not None else np.zeros(HIST_SIZE, np.float32)

        self.db.add_clothing_entry(
            person_id  = person_id,
            torso_hist = torso,
            legs_hist  = legs,
            shoes_hist = shoes,
            label      = label,
        )
        self._last_saved[person_id] = now

    def identify(self, zones: dict
                 ) -> tuple[Optional[int], Optional[str], float]:
        """Attempt to identify a person from clothing alone."""
        return match_wardrobe(self.db, zones)

    def get_description(self, zones: dict) -> str:
        """Return human-readable clothing description."""
        return describe_zones(zones)

    def draw_zones(self, frame: np.ndarray, zones: dict) -> np.ndarray:
        """Draw zone rectangles on frame for debug/demo."""
        colors = {
            "torso": (0, 200, 255),
            "legs":  (0, 255, 150),
            "shoes": (200, 150, 0),
        }
        for name, rect in zones.get("zone_rects", {}).items():
            x, y, w, h = rect
            color = colors.get(name, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            cv2.putText(frame, name, (x+2, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from db import NovaDB

    db    = NovaDB()
    layer = ClothingLayer(db)

    print("[ClothingLayer] Live webcam test.")
    print("  Press S to save current clothing to wardrobe for person_id=1")
    print("  Press M to attempt clothing match")
    print("  Press Q to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    # Simple face detector for testing
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        zones = None
        for (x, y, w, h) in faces[:1]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
            zones = layer.process(frame, (x, y, w, h))
            layer.draw_zones(frame, zones)
            desc = layer.get_description(zones)
            cv2.putText(frame, desc, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("ClothingLayer Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s') and zones:
            layer.save_to_wardrobe(1, zones, label="manual_test")
            print("[SAVED] Clothing saved for person_id=1")
        elif key == ord('m') and zones:
            pid, name, score = layer.identify(zones)
            if pid:
                print(f"[MATCH] {name} (score={score:.3f})")
            else:
                print(f"[NO MATCH] Best score={score:.3f}")

    cap.release()
    cv2.destroyAllWindows()
