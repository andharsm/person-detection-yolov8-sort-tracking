import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import streamlit as st
import tempfile

# Load model
MODEL_PATH = 'person_detection_yolov8/models/best.pt'
model = YOLO(MODEL_PATH)


VIDEO_PATH = 'person_detection_yolov8/videos/person in public area.mp4'

# SORT tracker
tracker = Sort(max_age=20, min_hits=5)

def process_frame(frame, recorded_redzone):  # Use a set to store unique person IDs
    results = model(frame)
    person_count = 0
    redzone_person_count = 0

    # redzone coordinates
    redzone_top_left = (250, 180)
    redzone_bottom_right = (450, 300)

    # List detected boxes
    dets = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            conf = box.conf.item()
            cls = box.cls
            label = model.names[int(cls)]

            if label == 'people':
                person_count += 1
                dets.append([x1, y1, x2, y2, conf])

    # Update tracker detections
    if len(dets) > 0:
        trackers = tracker.update(np.array(dets))
    else:
        trackers = []

    for d in trackers:
        x1, y1, x2, y2, track_id = d
        box_color = (11, 102, 35)

        # Check bounding inside the redzone
        if ((x1 >= redzone_top_left[0] and x1 <= redzone_bottom_right[0] and y2 <= redzone_bottom_right[1] and y2 >= redzone_top_left[1])
                or (x2 >= redzone_top_left[0] and x2 <= redzone_bottom_right[0] and y2 <= redzone_bottom_right[1] and y2 >= redzone_top_left[1])):
            box_color = (0, 0, 255)
            redzone_person_count += 1
            recorded_redzone.add(track_id)

        # bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # detail label
    frame_height, frame_width, _ = frame.shape
    label_text = f'person in frame: {person_count}'
    cv2.putText(frame, label_text, (20, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 102, 35), 2)

    label_text = f'person in red zone: {redzone_person_count}'
    cv2.putText(frame, label_text, (20, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    label_text = f'recorded in red zone: {len(recorded_redzone)}'
    cv2.putText(frame, label_text, (20, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # redzone
    cv2.rectangle(frame, redzone_top_left, redzone_bottom_right, (0, 0, 255), 2)  # Red rectangle

    return frame

def display_demo():
    st.write('## Demo Projek')

    st.write('### Video Original')
    st.video('person_detection_yolov8/videos/person in public area.mp4')

    if st.button('Apply Model'):
        st.write('### Proses Detection')
        cap = cv2.VideoCapture(VIDEO_PATH)
        stframe = st.empty()
        recorded_redzone = set()

        # Temp file to store the output video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = tfile.name

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = process_frame(frame, recorded_redzone)

            # Write the processed frame to the output video
            out.write(processed_frame)

            # Convert the frame to RGB
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display
            stframe.image(processed_frame, channels="RGB")

        cap.release()
        out.release()

        st.success("Pemrosesan video selesai!")

        st.write('### Download Video')
        with open(output_path, "rb") as video_file:
            st.download_button(label="Unduh Video", data=video_file, file_name="person_detection.mp4", mime="video/mp4")
