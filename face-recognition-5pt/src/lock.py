"""
Face Locking module (Term-02 Week-04).
Locks onto one enrolled identity, tracks that face with temporal smoothing,
and records actions (moved left/right, blink, smile) to a history file.

Enhanced Features:
- Robust face tracking with temporal filtering
- Improved action detection (blink, smile, movement)
- Better performance with frame skipping
- Visual feedback with action indicators
"""

import sys
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

from . import config
from .haar_5pt import HaarMediaPipeFaceDetector, FaceDetection
from .align import FaceAligner
from .embed import ArcFaceEmbedder


# ============================================================
# Face Tracker (smooth bounding box)
# ============================================================
class FaceTracker:
    def __init__(self, buffer_size=5):
        self.buf = deque(maxlen=buffer_size)

    def update(self, x1, y1, x2, y2):
        self.buf.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        avg = np.mean(self.buf, axis=0)
        return avg.astype(int)

    def reset(self):
        self.buf.clear()


# ============================================================
# Action Detector (movement / blink / smile)
# ============================================================
class ActionDetector:
    def __init__(self):
        self.prev_center_x = None
        self.center_hist = deque(maxlen=8)

        self.ear_hist = deque(maxlen=8)
        self.mouth_hist = deque(maxlen=20)

        self.last_action = {}
        self.blink_state = False
        self.smile_state = False

        self.visual_actions = deque(maxlen=5)  # (text, expiry_time)

    def _cooldown(self, key, frame_idx):
        last = self.last_action.get(key, -999)
        if frame_idx - last >= config.LOCK_ACTION_COOLDOWN_FRAMES:
            self.last_action[key] = frame_idx
            return True
        return False

    def detect(self, landmarks, face, frame_idx, W, H):
        actions = []

        # ---------------- Movement ----------------
        cx = (face.x1 + face.x2) / 2
        self.center_hist.append(cx)

        if len(self.center_hist) >= 6:
            dx = self.center_hist[-1] - self.center_hist[0]

            if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
                if self._cooldown("left", frame_idx):
                    actions.append(("face_moved_left", "⬅ LEFT"))
            elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
                if self._cooldown("right", frame_idx):
                    actions.append(("face_moved_right", "➡ RIGHT"))

        # ---------------- Eye blink ----------------
        ear_l = _ear_from_landmarks(
            landmarks, config.LOCK_EAR_LEFT_INDICES, W, H
        )
        ear_r = _ear_from_landmarks(
            landmarks, config.LOCK_EAR_RIGHT_INDICES, W, H
        )
        ear = (ear_l + ear_r) / 2
        self.ear_hist.append(ear)
        ear_avg = np.mean(self.ear_hist)

        blinking = ear_avg < config.LOCK_EAR_BLINK_THRESHOLD

        if blinking and not self.blink_state:
            if self._cooldown("blink", frame_idx):
                actions.append(("eye_blink", "👁 BLINK"))
            self.blink_state = True
        elif not blinking:
            self.blink_state = False

        # ---------------- Smile ----------------
        mouth_w = _mouth_width_from_landmarks(
            landmarks,
            config.LOCK_MOUTH_LEFT_INDEX,
            config.LOCK_MOUTH_RIGHT_INDEX,
            W, H,
        )
        self.mouth_hist.append(mouth_w)

        if len(self.mouth_hist) >= 15:
            baseline = np.median(self.mouth_hist)
            smiling = mouth_w >= baseline * config.LOCK_SMILE_MOUTH_RATIO

            if smiling and not self.smile_state:
                if self._cooldown("smile", frame_idx):
                    actions.append(("smile", "😊 SMILE"))
                self.smile_state = True
            elif not smiling:
                self.smile_state = False

        # store for overlay
        now = time.time()
        for _, txt in actions:
            self.visual_actions.append((txt, now + 1.2))

        return actions

    def draw_actions(self, vis, x, y):
        now = time.time()
        dy = 0
        for txt, expiry in list(self.visual_actions):
            if now <= expiry:
                cv2.putText(
                    vis, txt, (x, y - 20 - dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2
                )
                dy += 22

    def reset(self):
        self.prev_center_x = None
        self.center_hist.clear()
        self.ear_hist.clear()
        self.mouth_hist.clear()
        self.last_action.clear()
        self.visual_actions.clear()
        self.blink_state = False
        self.smile_state = False


# ============================================================
# Helpers
# ============================================================
def load_database():
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def cosine_distance(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return 1.0 - float(np.dot(a, b))


def _ear_from_landmarks(lms, idxs, W, H):
    pts = [(lms[i].x * W, lms[i].y * H) for i in idxs]
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    return (
        np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    ) / (2 * np.linalg.norm(p1 - p4) + 1e-6)


def _mouth_width_from_landmarks(lms, li, ri, W, H):
    l, r = lms[li], lms[ri]
    return np.hypot((r.x - l.x) * W, (r.y - l.y) * H)


# ============================================================
# MAIN
# ============================================================
def main():
    db = load_database()
    if not db:
        print("No enrolled identities.")
        return

    names = sorted(db.keys())
    print("Enrolled:", names)
    lock_identity = input("Lock identity: ").strip()
    if lock_identity not in db:
        print("Invalid identity.")
        return

    lock_idx = names.index(lock_identity)
    embeddings = np.stack([db[n] for n in names])

    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)

    face_tracker = FaceTracker()
    action_detector = ActionDetector()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera error")
        return

    mp_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    locked = False
    fail_count = 0
    frame_idx = 0
    threshold = config.DEFAULT_DISTANCE_THRESHOLD

    print("🔍 Searching...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        vis = frame.copy()
        H, W = frame.shape[:2]

        faces = detector.detect(frame)

        matched_face = None
        best_dist = 1.0
        
        # Process all faces and identify them
        face_identities = []  # (face, identity_name, distance, is_locked_target)
        
        for face in faces:
            aligned, _ = aligner.align(frame, face.landmarks)
            emb, _ = embedder.embed(aligned)
            
            # Find best match across all enrolled identities
            distances = [cosine_distance(emb, embeddings[i]) for i in range(len(embeddings))]
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            
            if min_dist <= threshold:
                identity = names[min_dist_idx]
                face_identities.append((face, identity, min_dist, identity == lock_identity))
                
                # Track if this is the locked identity
                if identity == lock_identity and min_dist < best_dist:
                    best_dist = min_dist
                    matched_face = face
            else:
                # Unknown face
                face_identities.append((face, "UNKNOWN", min_dist, False))

        # Handle locked state and track actions
        tracked_box = None
        if matched_face and best_dist <= threshold:
            fail_count = 0
            if not locked:
                locked = True
                face_tracker.reset()
                action_detector.reset()
                print("🔒 LOCKED")

            # Track and detect actions only for locked face
            tracked_box = face_tracker.update(
                matched_face.x1,
                matched_face.y1,
                matched_face.x2,
                matched_face.y2,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_mesh.process(rgb)

            if res.multi_face_landmarks:
                actions = action_detector.detect(
                    res.multi_face_landmarks[0].landmark,
                    matched_face, frame_idx, W, H
                )
        else:
            fail_count += 1
            if fail_count >= config.LOCK_RELEASE_FRAMES:
                if locked:
                    print("🔓 UNLOCKED")
                locked = False

        # Draw all faces with appropriate labels
        for face, identity, dist, is_target in face_identities:
            if is_target and locked:
                # Locked target: green box with tracker smoothing and actions
                x1, y1, x2, y2 = tracked_box
                color = (0, 255, 0)
                label = f"🔒 {identity}"
                thickness = 3
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
                
                # Draw actions only for locked target
                action_detector.draw_actions(vis, x1, y1)
                
            elif identity == "UNKNOWN":
                # Unknown face: red box
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                color = (0, 0, 255)
                label = "UNKNOWN"
                thickness = 2
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
            else:
                # Other known identity: just show name, no box
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                cv2.putText(
                    vis, identity,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 165, 0), 2  # Orange text
                )

        cv2.imshow("Face Locking", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
