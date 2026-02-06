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


class FaceTracker:
    """
    Smooths face tracking with temporal filtering.
    Prevents jittery bounding box movements.
    """
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.x1_buffer = deque(maxlen=buffer_size)
        self.y1_buffer = deque(maxlen=buffer_size)
        self.x2_buffer = deque(maxlen=buffer_size)
        self.y2_buffer = deque(maxlen=buffer_size)
    
    def update(self, x1, y1, x2, y2):
        """Add new bounding box and return smoothed version."""
        self.x1_buffer.append(float(x1))
        self.y1_buffer.append(float(y1))
        self.x2_buffer.append(float(x2))
        self.y2_buffer.append(float(y2))
        
        smooth_x1 = int(np.mean(self.x1_buffer))
        smooth_y1 = int(np.mean(self.y1_buffer))
        smooth_x2 = int(np.mean(self.x2_buffer))
        smooth_y2 = int(np.mean(self.y2_buffer))
        
        return smooth_x1, smooth_y1, smooth_x2, smooth_y2
    
    def reset(self):
        """Reset buffers (for new lock)."""
        self.x1_buffer.clear()
        self.y1_buffer.clear()
        self.x2_buffer.clear()
        self.y2_buffer.clear()


class ActionDetector:
    """
    Robust action detection with state machine approach.
    Detects: face_moved_left, face_moved_right, eye_blink, smile.
    """
    def __init__(self):
        self.last_action_frame = {}
        self.prev_center_x = None
        self.baseline_mouth_width = None
        self.mouth_width_samples = deque(maxlen=30)
        self.ear_samples = deque(maxlen=10)
        self.blink_state = False  # Track if currently in blink
        self.smile_state = False  # Track if currently smiling
    
    def detect(self, frame, matched_face, frame_idx, W, H):
        """Detect actions and return list of (action_type, description) tuples."""
        actions = []
        cooldown = config.LOCK_ACTION_COOLDOWN_FRAMES
        
        center_x = (matched_face.x1 + matched_face.x2) / 2.0
        
        # Movement detection (left/right)
        if self.prev_center_x is not None:
            dx = center_x - self.prev_center_x
            if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
                if frame_idx - self.last_action_frame.get("face_moved_left", -999) >= cooldown:
                    actions.append(("face_moved_left", "↶ moved left"))
                    self.last_action_frame["face_moved_left"] = frame_idx
            elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
                if frame_idx - self.last_action_frame.get("face_moved_right", -999) >= cooldown:
                    actions.append(("face_moved_right", "↷ moved right"))
                    self.last_action_frame["face_moved_right"] = frame_idx
        
        # Get full landmarks for eye and mouth detection
        landmarks_list = _get_full_landmarks(frame)
        if landmarks_list is not None:
            # Eye blink detection
            ear_left = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_LEFT_INDICES, W, H)
            ear_right = _ear_from_landmarks(landmarks_list, config.LOCK_EAR_RIGHT_INDICES, W, H)
            ear = (ear_left + ear_right) / 2.0
            
            self.ear_samples.append(ear)
            ear_avg = np.mean(list(self.ear_samples)) if self.ear_samples else ear
            
            # State machine for blink detection
            is_blinking = ear < config.LOCK_EAR_BLINK_THRESHOLD * 0.95
            if is_blinking and not self.blink_state:
                if frame_idx - self.last_action_frame.get("eye_blink", -999) >= cooldown:
                    actions.append(("eye_blink", "👁️ blinked"))
                    self.last_action_frame["eye_blink"] = frame_idx
                self.blink_state = True
            elif not is_blinking:
                self.blink_state = False
            
            # Smile detection
            mouth_width = _mouth_width_from_landmarks(
                landmarks_list, config.LOCK_MOUTH_LEFT_INDEX,
                config.LOCK_MOUTH_RIGHT_INDEX, W, H
            )
            self.mouth_width_samples.append(mouth_width)
            
            if self.baseline_mouth_width is None and len(self.mouth_width_samples) >= 15:
                self.baseline_mouth_width = float(np.median(self.mouth_width_samples))
            
            # State machine for smile detection
            is_smiling = (
                self.baseline_mouth_width is not None and
                self.baseline_mouth_width > 1.0 and
                mouth_width >= self.baseline_mouth_width * config.LOCK_SMILE_MOUTH_RATIO
            )
            if is_smiling and not self.smile_state:
                if frame_idx - self.last_action_frame.get("smile", -999) >= cooldown:
                    actions.append(("smile", "😊 smiled"))
                    self.last_action_frame["smile"] = frame_idx
                self.smile_state = True
            elif not is_smiling:
                self.smile_state = False
        
        self.prev_center_x = center_x
        return actions
    
    def reset(self):
        """Reset state for new lock."""
        self.last_action_frame.clear()
        self.prev_center_x = None
        self.baseline_mouth_width = None
        self.mouth_width_samples.clear()
        self.ear_samples.clear()
        self.blink_state = False
        self.smile_state = False


def load_database():
    """Load enrolled face database."""
    if not config.DB_NPZ_PATH.exists():
        return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def cosine_distance(a, b):
    """Compute cosine distance between two vectors (optimized)."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return 1.0 - float(np.dot(a, b))


def _ear_from_landmarks(landmarks_list, indices, W, H):
    """Eye Aspect Ratio from 6 landmark indices (vertical/horizontal ratio)."""
    pts = []
    for i in indices:
        lm = landmarks_list[i]
        pts.append((lm.x * W, lm.y * H))
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.5
    return (v1 + v2) / (2.0 * h)


def _mouth_width_from_landmarks(landmarks_list, left_idx, right_idx, W, H):
    """Mouth width in pixels."""
    l = landmarks_list[left_idx]
    r = landmarks_list[right_idx]
    return np.hypot((r.x - l.x) * W, (r.y - l.y) * H)


def _get_full_landmarks(frame):
    """Run MediaPipe Face Mesh on frame; return first face landmark list or None."""
    if mp is None:
        return None
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results = mesh.process(rgb)
    mesh.close()
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


def main():
    """Face Locking: select one identity, lock onto that face, track and record actions."""
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False

    names = sorted(db.keys())
    print("\nEnrolled identities:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0] if names else ""
    if not choice:
        choice = names[0] if names else ""
    if choice not in db:
        print(f"ERROR: '{choice}' not in database. Choose from: {names}")
        return False
    lock_identity = choice
    print("Will lock onto:", lock_identity)

    config.ensure_dirs()
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
    lock_idx = names.index(lock_identity)
    threshold = config.DEFAULT_DISTANCE_THRESHOLD

    # Initialize new components
    face_tracker = FaceTracker(buffer_size=5)
    action_detector = ActionDetector()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False

    locked = False
    fail_count = 0
    history_file = None
    history_path = None
    frame_idx = 0
    process_frame_interval = 2  # Process every 2nd frame for performance

    print("\nFace Locking - When the selected face appears, system will lock. q=Quit")
    print("Controls: q=quit, +/-=threshold, r=reload")

    t0 = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            frame_count += 1

            vis = frame.copy()
            H, W = frame.shape[:2]
            matched_face = None  # Initialize for use in all paths
            best_dist = 1.0
            
            # Performance: Skip detection on some frames when unlocked
            should_detect = (frame_idx % process_frame_interval == 0) or not locked
            faces = detector.detect(frame) if should_detect else []

            if not locked:
                # Search for target identity
                for face in faces:
                    aligned, _ = aligner.align(frame, face.landmarks)
                    query_emb, _ = embedder.embed(aligned)
                    dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                    best_idx = int(np.argmin(dists))
                    best_dist = dists[best_idx]
                    
                    # Lock acquired!
                    if best_idx == lock_idx and best_dist <= threshold:
                        locked = True
                        fail_count = 0
                        face_tracker.reset()
                        action_detector.reset()
                        
                        # Create history file
                        ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
                        safe_name = lock_identity.replace(" ", "_").lower()
                        history_path = config.HISTORY_DIR / (safe_name + "_history_" + ts + ".txt")
                        history_file = open(history_path, "w", encoding="utf-8")
                        history_file.write("# Face Lock history: " + lock_identity + "\n")
                        history_file.write("# Started: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
                        history_file.write("# Format: timestamp  action_type  description\n# ---\n")
                        history_file.flush()
                        print(f"🔒 LOCKED onto {lock_identity} | Recording to {history_path.name}")
                        break

            else:
                # Track locked identity
                matched_face = None
                best_dist = 1.0
                
                for face in faces:
                    aligned, _ = aligner.align(frame, face.landmarks)
                    query_emb, _ = embedder.embed(aligned)
                    dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                    idx = int(np.argmin(dists))
                    d = dists[idx]
                    
                    # Match found!
                    if idx == lock_idx and d <= threshold:
                        matched_face = face
                        best_dist = d
                        fail_count = 0
                        break
                    
                    # Fallback: single face, close to threshold
                    if len(faces) == 1 and d < 0.5:
                        matched_face = face
                        best_dist = d
                        fail_count += 1
                        if fail_count > 5:
                            matched_face = None
                        break

                if matched_face is None:
                    fail_count += 1
                    if fail_count >= config.LOCK_RELEASE_FRAMES:
                        locked = False
                        if history_file:
                            history_file.close()
                            history_file = None
                        print(f"🔓 Lock released (no face for {config.LOCK_RELEASE_FRAMES} frames)")
                else:
                    # Smooth face tracking
                    smooth_x1, smooth_y1, smooth_x2, smooth_y2 = face_tracker.update(
                        matched_face.x1, matched_face.y1, matched_face.x2, matched_face.y2
                    )
                    
                    # Detect actions with new detector
                    action_list = action_detector.detect(frame, matched_face, frame_idx, W, H)
                    
                    # Record actions to file
                    ts = time.time()
                    for action_type, desc in action_list:
                        line = "%.2f  %s  %s\n" % (ts, action_type, desc)
                        if history_file:
                            history_file.write(line)
                            history_file.flush()
                        print(f"  Action: {desc}")
                    
                    # Store smoothed position for next draw
                    matched_face.x1 = smooth_x1
                    matched_face.y1 = smooth_y1
                    matched_face.x2 = smooth_x2
                    matched_face.y2 = smooth_y2
                    
                    # Draw locked face with enhanced visuals
                    cv2.rectangle(vis, (matched_face.x1, matched_face.y1), (matched_face.x2, matched_face.y2), (0, 255, 0), 4)
                    cv2.putText(vis, f"🔒 {lock_identity}", (matched_face.x1, max(0, matched_face.y1 - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(vis, f"dist={best_dist:.3f}", (matched_face.x1, matched_face.y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Draw other detected faces (not the lock target)
            if locked and matched_face is not None:
                for face in faces:
                    # Skip the locked face
                    if face.x1 == matched_face.x1 and face.y1 == matched_face.y1:
                        continue
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 0, 255), 2)
                    cv2.putText(vis, "Other", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            elif not locked:
                # Show searching status
                for face in faces:
                    cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                    cv2.putText(vis, "Searching...", (face.x1, face.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Enhanced header
            status = "🔒 LOCKED" if locked else "🔍 Searching..."
            header = f"Target: {lock_identity} | {status} | FPS: {fps:.1f}"
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(vis, "q=quit | +/-=threshold | r=reload", (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Face Locking", vis)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                threshold = min(1.0, threshold + 0.01)
                print(f"Threshold increased to {threshold:.2f}")
            elif key == ord("-"):
                threshold = max(0.0, threshold - 0.01)
                print(f"Threshold decreased to {threshold:.2f}")
            elif key == ord("r"):
                db = load_database()
                names = sorted(db.keys())
                embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
                print(f"Reloaded {len(db)} identities from database")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if history_file:
            history_file.close()
        if history_path:
            print(f"✓ History saved to: {history_path}")



if __name__ == "__main__":
    main()
