"""
Camera module: validates video capture and FPS.
First sanity check for the entire pipeline.
"""

import sys
import cv2
from pathlib import Path

def main():
    """
    Open webcam and display live video with FPS counter.
    Press 'q' to exit.
    """
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {camera_index}.")
        print("Troubleshooting:")
        print("  - macOS: System Settings > Privacy & Security > Camera")
        print("  - Windows/Linux: Ensure no other app is using the camera")
        print("  - Try different camera indices: 0, 1, 2")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera opened successfully.")
    print("  Press 'q' to exit.")
    
    frame_count = 0
    import time
    t0 = time.time()
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break
            
            # Calculate FPS every second
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()
            
            # Draw FPS
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                frame, "Press 'q' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("Camera Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Camera test passed. Pipeline ready to proceed.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
