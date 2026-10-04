# Smart Attendance System

Face-recognition attendance using **OpenCV** and **face_recognition**.

Mark attendance from a live camera feed; recognised students are logged with timestamp.

## Features

- Load known faces from `known_faces/<StudentName>/photo.jpg`
- Live webcam recognition
- Attendance log CSV (`name`, `timestamp`)
- Demo mode without camera (for CI / quick check)

## Setup

```bash
pip install -r requirements.txt
```

> Note: `face_recognition` needs `dlib`. On some systems install build tools first.

### Add students

```
known_faces/
  Gowtham/
    photo1.jpg
  Arun/
    photo1.jpg
```

## Run

```bash
# Live camera
python attendance.py --camera

# Demo (no camera)
python attendance.py --demo
```

Press **Q** to quit the camera window.

## Flow

1. Capture frame from webcam  
2. Detect faces  
3. Match against known encodings  
4. Log name + timestamp to `attendance_log.csv`

## Author

**Gowtham V** · [github.com/gowtham-v08](https://github.com/gowtham-v08) · [Portfolio](https://gowtham-v.vercel.app)
