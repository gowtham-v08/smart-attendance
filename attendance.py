"""
Smart Attendance System
Face recognition attendance using OpenCV + face_recognition (when available).
Falls back to a clear demo mode if heavy deps are missing.
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import face_recognition
    import numpy as np

    HAS_CV = True
except ImportError:
    HAS_CV = False


DATA_DIR = Path("known_faces")
LOG_FILE = Path("attendance_log.csv")


def ensure_log():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["name", "timestamp"])


def mark_attendance(name: str):
    ensure_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, ts])
    print(f"[ATTENDANCE] {name} @ {ts}")


def load_known_faces():
    """Load encodings from known_faces/<name>/*.jpg"""
    encodings = []
    names = []
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return encodings, names

    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")):
            image = face_recognition.load_image_file(str(img_path))
            faces = face_recognition.face_encodings(image)
            if faces:
                encodings.append(faces[0])
                names.append(name)
                print(f"Loaded encoding for {name} from {img_path.name}")
    return encodings, names


def run_camera():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No known faces found.")
        print(f"Add photos under: {DATA_DIR}/<StudentName>/photo.jpg")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    already_marked = set()
    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), encoding in zip(locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                if name not in already_marked:
                    mark_attendance(name)
                    already_marked.add(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            cv2.putText(frame, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow("Smart Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_mode():
    print("=== Smart Attendance · DEMO MODE ===")
    print("OpenCV / face_recognition not installed in this environment.")
    print()
    print("How the real system works:")
    print("1. Put student photos in known_faces/<Name>/photo.jpg")
    print("2. Run: python attendance.py --camera")
    print("3. Camera opens; recognised students are logged to attendance_log.csv")
    print()
    # Simulate a few marks so the log file exists for demos
    for student in ["Gowtham", "Arun", "Priya"]:
        mark_attendance(student)
    print(f"Sample log written to {LOG_FILE.resolve()}")
    print("Install deps and use a webcam for live recognition.")


def main():
    parser = argparse.ArgumentParser(description="Smart Attendance System")
    parser.add_argument("--camera", action="store_true", help="Run live camera recognition")
    parser.add_argument("--demo", action="store_true", help="Run demo without camera")
    args = parser.parse_args()

    if args.camera:
        if not HAS_CV:
            print("Install: pip install opencv-python face_recognition numpy")
            return
        run_camera()
    else:
        demo_mode()


if __name__ == "__main__":
    main()
