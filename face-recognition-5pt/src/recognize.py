"""
Live face recognition module.
Real-time face matching against enrolled database.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np


from . import config
from .haar_5pt import HaarMediaPipeFaceDetector
from .align import FaceAligner
from .embed import ArcFaceEmbedder


def load_database():
    """Load enrolled face database."""
    if not config.DB_NPZ_PATH.exists():
        print("ERROR: Database not found. Run enrollment first.")
        return {}
    
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance."""
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    similarity = float(np.dot(a, b))
    return 1.0 - similarity


def main():
    """Live recognition pipeline."""
    db = load_database()
    
    if not db:
        print("ERROR: No enrolled identities found. Run enrollment first.")
        return False
    
    print(f"✓ Loaded {len(db)} enrolled identities")
    
    detector = HaarMediaPipeFaceDetector(min_size=config.HAAR_MIN_SIZE)
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    
    # Pre-stack embeddings for fast matching
    names = sorted(db.keys())
    embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False
    
    threshold = config.DEFAULT_DISTANCE_THRESHOLD
    
    print("\nLive Recognition")
    print("Controls:")
    print("  q  - Quit")
    print("  r  - Reload database")
    print("  +  - Increase threshold (more accepts)")
    print("  -  - Decrease threshold (fewer accepts)")
    
    try:
        import time
        t0 = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            
            vis = frame.copy()
            faces = detector.detect(frame)
            
            for face_idx, face in enumerate(faces):
                # Draw bbox + landmarks
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                
                for (x, y) in face.landmarks.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Align + embed
                aligned, _ = aligner.align(frame, face.landmarks)
                query_emb, _ = embedder.embed(aligned)
                
                # Match
                dists = np.array([cosine_distance(query_emb, embeddings_matrix[i]) for i in range(len(names))])
                best_idx = int(np.argmin(dists))
                best_dist = dists[best_idx]
                
                # Decision
                if best_dist <= threshold:
                    name = names[best_idx]
                    confidence = 1.0 - best_dist
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    confidence = 0
                    color = (0, 0, 255)
                
                # Draw label
                cv2.putText(
                    vis, f"{name} ({best_dist:.3f})", (face.x1, max(0, face.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                
                # Draw confidence bar
                bar_w = 100
                bar_h = 20
                bar_x = face.x1
                bar_y = face.y1 - 35
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
                if best_dist <= threshold:
                    filled_w = int(bar_w * confidence)
                    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), color, -1)
            
            # Header
            header = f"Threshold: {threshold:.2f} | IDs: {len(names)} | FPS: {fps:.1f}"
            cv2.putText(
                vis, header, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Controls hint
            cv2.putText(
                vis, "q=quit, r=reload, +/-=threshold", (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("Live Recognition", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                db = load_database()
                names = sorted(db.keys())
                embeddings_matrix = np.stack([db[n].reshape(-1) for n in names], axis=0)
                print(f"✓ Reloaded {len(db)} identities")
            elif key in (ord("+"), ord("=")):
                threshold = min(1.0, threshold + 0.01)
                print(f"Threshold: {threshold:.2f}")
            elif key == ord("-"):
                threshold = max(0.0, threshold - 0.01)
                print(f"Threshold: {threshold:.2f}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Recognition ended.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
