import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10, '10Ag': 0.1, '50Ag': 0.5}

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = next(model.track(
            source=frame,
            persist=True,
            stream=True,
            tracker="botsort.yaml",
            conf=0.5,
            verbose=False,
            vid_stride=2
        ))
        annotated_frame = result.plot()
        total_sum = 0
        boxes = result.boxes
        if boxes.id is not None:
            for i in range(len(boxes.id)):
                track_id = int(boxes.id[i].item())
                cls_id = int(boxes.cls[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                if w < 30 or h < 30:
                    continue
                label = model.names.get(cls_id, "Unknown")
                value = value_map.get(label, 0)
                total_sum += value
                cv2.putText(
                    annotated_frame,
                    f"ID:{track_id}",
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
        cv2.putText(
            annotated_frame,
            f"Total: {total_sum:.2f} ILS",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        # המרה ל-JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


def process_frame(frame_np):
    # frame_np: numpy array, RGB
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    result = next(model.track(
        source=frame_bgr,
        persist=True,
        stream=True,
        tracker="botsort.yaml",
        conf=0.5,
        verbose=False,
        vid_stride=2
    ))
    annotated_frame = result.plot()
    total_sum = 0
    boxes = result.boxes
    if boxes.id is not None:
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            if w < 30 or h < 30:
                continue
            label = model.names.get(cls_id, "Unknown")
            value = value_map.get(label, 0)
            total_sum += value
            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (int(xyxy[0]), int(xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
    cv2.putText(
        annotated_frame,
        f"Total: {total_sum:.2f} ILS",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    return annotated_frame 