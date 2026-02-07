import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(f"Camera {i}:", "OK" if cap.isOpened() else "FAIL")
    cap.release()
