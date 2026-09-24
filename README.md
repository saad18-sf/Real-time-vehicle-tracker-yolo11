# Real-Time Vehicle Tracking & Bidirectional Traffic Counting

A high-performance computer vision system for bidirectional traffic flow monitoring, vehicle classification, and zone-based counting using Ultralytics YOLO11, ByteTrack, and Roboflow Supervision.

---

## Features

- **State-of-the-Art Object Detection:** Powered by YOLO11 for accurate multi-class vehicle detection (`car`, `bus`, `truck`, `motorbike`).
- **Persistent Multi-Object Tracking:** Uses ByteTrack to maintain unique object IDs across video frames.
- **Track Smoothing:** Integrates Supervision's `DetectionsSmoother` to reduce bounding box jitter and flickering.
- **Bidirectional Zone Counting:** Scaled polygonal regions calculate directional flow (UP / DOWN) and track total volume using bottom-center anchor points.
- **Hardware Acceleration:** Automatic fallback supporting CUDA with half-precision (FP16) inference or standard CPU execution.
- **Real-Time Visual Dashboard:** Live HUD displaying total detections, directional counts, and bounding box tracks.

---

## Directory Structure

```text
.
├── car.py                     # Main tracking and video analysis pipeline
├── requirements.txt           # Project dependencies
├── .gitignore                 # Excluded environments, video files, and weights
└── README.md                  # Project documentation
