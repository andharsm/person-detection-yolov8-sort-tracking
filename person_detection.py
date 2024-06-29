import cv2
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Mengimpor SORT dari direktori lokal

MODEL_PATH = os.path.abspath('person_detection_yolov8/models/best.pt')
PERSON_IN_PUBLIC_AREA_VIDEO_PATH = os.path.abspath('person_detection_yolov8/videos/person in public area.mp4')

# Load model
model = YOLO(MODEL_PATH)

recorded_redzone = set()  # Use a set to store unique person IDs

# Load video
cap = cv2.VideoCapture(PERSON_IN_PUBLIC_AREA_VIDEO_PATH)

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0
    redzone_person_count = 0

    # Define redzone coordinates
    redzone_top_left = (250, 180)
    redzone_bottom_right = (450, 300)

    # List to hold detected boxes in the format (x1, y1, x2, y2, conf)
    dets = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            conf = box.conf.item()  # Make sure conf is a scalar
            cls = box.cls
            label = model.names[int(cls)]

            if label == 'people':
                person_count += 1
                dets.append([x1, y1, x2, y2, conf])

    # Update tracker with detections
    if len(dets) > 0:
        trackers = tracker.update(np.array(dets))
    else:
        trackers = []

    for d in trackers:
        x1, y1, x2, y2, track_id = d
        box_color = (11, 102, 35)

        # Check if the bounding box is inside the redzone
        if ((x1 >= redzone_top_left[0] and x1 <= redzone_bottom_right[0] and y2 <= redzone_bottom_right[1] and y2 >= redzone_top_left[1])
                or (x2 >= redzone_top_left[0] and x2 <= redzone_bottom_right[0] and y2 <= redzone_bottom_right[1] and y2 >= redzone_top_left[1])):
            box_color = (0, 0, 255)
            redzone_person_count += 1
            recorded_redzone.add(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Add label in the bottom right corner
    frame_height, frame_width, _ = frame.shape
    label_text = f'person in frame: {person_count}'
    cv2.putText(frame, label_text, (20, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 102, 35), 2)

    label_text = f'person in red zone: {redzone_person_count}'
    cv2.putText(frame, label_text, (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    label_text = f'recorded in red zone: {len(recorded_redzone)}'
    cv2.putText(frame, label_text, (20, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw rectangle redzone
    cv2.rectangle(frame, redzone_top_left, redzone_bottom_right, (0, 0, 255), 2)  # Red rectangle

    # Display the frame with detections
    cv2.imshow('Person Detection', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
