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
from datetime import datetime
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
# History Logger (save actions to file)
# ============================================================
class HistoryLogger:
    def __init__(self, identity_name, history_dir=None):
        """Initialize history logger for a specific identity."""
        self.identity_name = identity_name
        
        if history_dir is None:
            history_dir = config.HISTORY_DIR
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{identity_name}_history_{timestamp}.txt"
        self.history_file = self.history_dir / filename
        
        # Create history file with header
        self._create_header()
        print(f"📝 History logging to: {self.history_file}")
    
    def _create_header(self):
        """Create history file with header information."""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            f.write(f"# Face Lock history: {self.identity_name}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: timestamp  action_type  description\n")
            f.write("# ---\n")
            f.write("\n")
    
    def log_action(self, action_type, description=""):
        """Log an action to the history file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"{timestamp}  {action_type:20s}  {description}\n"
        
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"⚠ Failed to write to history: {e}")
    
    def log_lock_start(self):
        """Log when locking starts."""
        self.log_action("LOCK_START", f"Locked onto {self.identity_name}")
    
    def log_lock_release(self):
        """Log when lock is released."""
        self.log_action("LOCK_RELEASE", f"Released lock on {self.identity_name}")
    
    def log_multi_person(self, identities_list):
        """Log when multiple people are detected."""
        identities_str = ", ".join(identities_list)
        self.log_action("MULTI_PERSON", f"Multiple people: {identities_str}")


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

        if len(self.center_hist) >= 4:  # Reduced for more responsive detection
            dx = self.center_hist[-1] - self.center_hist[0]

            # Log movement continuously while moving (with cooldown)
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

        # Log blink continuously while blinking (with cooldown)
        if blinking:
            if not self.blink_state:
                # First blink detection
                if self._cooldown("blink", frame_idx):
                    actions.append(("eye_blink", "👁 BLINK"))
                self.blink_state = True
            else:
                # Continue logging while still blinking (with cooldown)
                if self._cooldown("blink", frame_idx):
                    actions.append(("eye_blink", "👁 BLINK"))
        else:
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

            # Log smile continuously while smiling (with cooldown)
            if smiling:
                if not self.smile_state:
                    # First smile detection
                    if self._cooldown("smile", frame_idx):
                        actions.append(("smile", "😊 SMILE"))
                    self.smile_state = True
                else:
                    # Continue logging while still smiling (with cooldown)
                    if self._cooldown("smile", frame_idx):
                        actions.append(("smile", "😊 SMILE"))
            else:
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
    # Multiple action detectors - one for each face identity
    action_detectors = {}  # {identity_name: ActionDetector}
    
    # History logger for the locked identity
    history_logger = HistoryLogger(lock_identity)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera error")
        return

    # Enable multi-face detection for MediaPipe
    mp_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,  # Support up to 5 faces simultaneously
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
        
        # Log multi-person detection
        if len(face_identities) > 1:
            identities_list = [f[1] for f in face_identities]
            identities_str = ", ".join(identities_list)
            if frame_idx % 30 == 0:  # Log every 30 frames to avoid spam
                print(f"👥 Multiple people detected: {identities_str}")
                history_logger.log_multi_person(identities_list)

        # Handle locked state and track actions
        tracked_box = None
        if matched_face and best_dist <= threshold:
            fail_count = 0
            if not locked:
                locked = True
                face_tracker.reset()
                print(f"🔒 LOCKED onto {lock_identity}")
                history_logger.log_lock_start()

            # Track locked face with smoothing
            tracked_box = face_tracker.update(
                matched_face.x1,
                matched_face.y1,
                matched_face.x2,
                matched_face.y2,
            )
        
        # Process behaviors for ALL detected faces
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_mesh.process(rgb)
        
        face_behaviors = {}  # {identity_name: actions_dict}
        
        if res.multi_face_landmarks and len(face_identities) > 0:
            # Match each MediaPipe mesh to a detected face by proximity
            for mp_landmarks in res.multi_face_landmarks:
                # Get center of MediaPipe mesh
                mp_x = int(mp_landmarks.landmark[1].x * W)
                mp_y = int(mp_landmarks.landmark[1].y * H)
                
                # Find closest detected face
                min_distance = float('inf')
                closest_face_identity = None
                closest_face = None
                
                for face, identity, dist, is_target in face_identities:
                    face_center_x = (face.x1 + face.x2) // 2
                    face_center_y = (face.y1 + face.y2) // 2
                    distance = ((mp_x - face_center_x)**2 + (mp_y - face_center_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_face_identity = identity
                        closest_face = face
                
                # Only process if mesh is close enough to a detected face (within face bounds)
                if min_distance < 150 and closest_face_identity:
                    # Create action detector for this identity if not exists
                    if closest_face_identity not in action_detectors:
                        action_detectors[closest_face_identity] = ActionDetector()
                    
                    # Detect actions for this face
                    actions = action_detectors[closest_face_identity].detect(
                        mp_landmarks.landmark,
                        closest_face, frame_idx, W, H
                    )
                    face_behaviors[closest_face_identity] = actions
                    
                    # Log actions for the locked identity
                    if closest_face_identity == lock_identity and actions:
                        for action_type, action_text in actions:
                            history_logger.log_action(action_type.upper(), action_text)
        else:
            fail_count += 1
            if fail_count >= config.LOCK_RELEASE_FRAMES:
                if locked:
                    print("🔓 UNLOCKED")
                    history_logger.log_lock_release()
                locked = False

        # Draw all faces with appropriate labels and behaviors
        for face, identity, dist, is_target in face_identities:
            x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
            
            if is_target and locked:
                # Locked target: green box with tracker smoothing
                x1, y1, x2, y2 = tracked_box
                color = (0, 255, 0)
                label = f"🔒 {identity} (TARGET)"
                thickness = 3
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
                
                # Draw actions for locked target
                if identity in face_behaviors:
                    action_detectors[identity].draw_actions(vis, x1, y1)
                
            elif identity == "UNKNOWN":
                # Unknown face: red box
                color = (0, 0, 255)
                label = "❓ UNKNOWN"
                thickness = 2
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
                
                # Show behaviors for unknown faces too (if detected)
                if identity in face_behaviors:
                    action_detectors[identity].draw_actions(vis, x1, y1)
            else:
                # Other known identity: blue box with behaviors
                color = (255, 165, 0)  # Orange
                label = f"👤 {identity}"
                thickness = 2
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2
                )
                
                # Draw actions for other known faces
                if identity in face_behaviors:
                    action_detectors[identity].draw_actions(vis, x1, y1)
        
        # Display multi-person detection status
        num_detected = len(face_identities)
        known_count = len([f for f in face_identities if f[1] != "UNKNOWN"])
        unknown_count = num_detected - known_count
        
        status_y = 30
        cv2.putText(vis, f"👥 Detected: {num_detected} people", 
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if num_detected > 0:
            status_y += 30
            cv2.putText(vis, f"✓ Known: {known_count}  ❌ Unknown: {unknown_count}", 
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if locked:
            status_y += 30
            cv2.putText(vis, f"🎯 Target: {lock_identity}", 
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Locking", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Log session end
    history_logger.log_action("SESSION_END", "Face locking session ended")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
