import cv2 as cv
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

# 1. Device selection & model loading
# Uses CUDA if available; otherwise falls back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"

# Switched to yolo11n.pt for real-time speed
model = YOLO("yolo11n.pt").to(device)

video_path = "cars_on_highway_5.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# 2. Tracking and class configuration
tracker = sv.ByteTrack(frame_rate=fps)
smoother = sv.DetectionsSmoother()
vehicle_classes = {"car", "motorbike", "bus", "truck"}
selected_classes = [
    cls_id
    for cls_id, class_name in model.names.items()
    if class_name in vehicle_classes
]

# 3. Counting zone setup
arr1 = np.array(
    [
        [761, 642],
        [1073, 642],
        [1070, 732],
        [968, 776],
        [872, 1038],
        [97, 1049],
    ],
    dtype=np.int32,
)
arr2 = np.array(
    [
        [1105, 639],
        [1402, 645],
        [1920, 959],
        [1920, 1080],
        [930, 1073],
        [991, 811],
        [1105, 755],
    ],
    dtype=np.int32,
)
zone_points = [np.floor(arr * 0.66).astype(np.int32) for arr in [arr1, arr2]]
zones = [sv.PolygonZone(points) for points in zone_points]
colors = sv.ColorPalette.from_hex(["#ef260e", "#07f921"])

# 4. Global annotators
thickness = sv.calculate_optimal_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale(
    resolution_wh=video_info.resolution_wh
)

box_annotator = sv.RoundBoxAnnotator(
    thickness=thickness, color_lookup=sv.ColorLookup.TRACK
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.TOP_CENTER,
    color_lookup=sv.ColorLookup.TRACK,
)
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=z,
        thickness=3,
        color=colors.by_idx(i),
        text_scale=1.5,
        text_thickness=2,
    )
    for i, z in enumerate(zones)
]

# 5. Tracking state
total_counts, crossed_ids = [], set()
counts_up, ids_up = [], set()
counts_down, ids_down = [], set()


def count_vehicle(ID, cx, cy):
  # Zone 0 (arr1, Left): Moving UP / Away
  if cv.pointPolygonTest(zone_points[0], (cx, cy), False) >= 0:
    if ID not in crossed_ids:
      total_counts.append(ID)
      crossed_ids.add(ID)
    if ID not in ids_up:
      counts_up.append(ID)
      ids_up.add(ID)

  # Zone 1 (arr2, Right): Moving DOWN / Toward
  elif cv.pointPolygonTest(zone_points[1], (cx, cy), False) >= 0:
    if ID not in crossed_ids:
      total_counts.append(ID)
      crossed_ids.add(ID)
    if ID not in ids_down:
      counts_down.append(ID)
      ids_down.add(ID)


def annotate_frame(frame, detections):
  for zone_annotator in zone_annotators:
    zone_annotator.annotate(frame)

  mask = (
      np.isin(detections.class_id, selected_classes)
      & (detections.tracker_id != None)  # noqa: E711
  )
  valid_dets = detections[mask]

  labels = [
      f"{model.names[cid]} #{tid}"
      for cid, tid in zip(valid_dets.class_id, valid_dets.tracker_id)
  ]
  frame = box_annotator.annotate(frame, valid_dets)
  frame = label_annotator.annotate(frame, valid_dets, labels=labels)

  bottom_centers = valid_dets.get_anchors_coordinates(
      anchor=sv.Position.BOTTOM_CENTER
  )
  for track_id, bottom_center in zip(valid_dets.tracker_id, bottom_centers):
    cx, cy = int(bottom_center[0]), int(bottom_center[1])
    cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)
    count_vehicle(track_id, cx, cy)

  # Stats dashboard overlay
  counter_labels = [
      f"COUNTS: {len(total_counts)}",
      f"UP: {len(counts_up)}",
      f"DOWN: {len(counts_down)}",
  ]
  count_colors = [(0, 0, 0), (6, 104, 2), (0, 0, 255)]
  cv.rectangle(frame, (0, 0), (320, 150), (255, 255, 255), cv.FILLED)
  for i, (label, color) in enumerate(zip(counter_labels, count_colors)):
    cv.putText(
        frame,
        label,
        (20, 45 + i * 42),
        cv.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
    )

  return frame


# 6. Video processing loop
cap = cv.VideoCapture(video_path)
output_path = "car_counter_yolo11n.mp4"
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # Fast inference: set imgsz=640, enable half-precision on GPU
  results = model(
      frame,
      verbose=False,
      imgsz=640,
      half=use_fp16,
      classes=selected_classes,
  )[0]

  detections = sv.Detections.from_ultralytics(results)
  detections = tracker.update_with_detections(detections)
  detections = smoother.update_with_detections(detections)

  for zone in zones:
    zone.trigger(detections)

  if detections.tracker_id is not None and len(detections.tracker_id) > 0:
    frame = annotate_frame(frame, detections)

  out.write(frame)
  cv.imshow("Video", frame)

  if cv.waitKey(1) & 0xFF == ord("p"):
    break

cap.release()
out.release()
cv.destroyAllWindows()