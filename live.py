from ultralytics import YOLO
import cv2

# טען את המודל שלך
model = YOLO("best.pt")

# ערכים כספיים לפי תוויות
value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10}

# שמירה על track_ids שכבר נספרו
seen_ids = set()
total_sum = 0

# פתיחת מצלמה
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # שליפת תוצאה בודדת מה-generator
    result = next(model.track(
        source=frame,
        persist=True,
        stream=True,  # חשוב! אחרת זה לא ייתן תוצאה כ־generator
        tracker="botsort.yaml",
        conf=0.5,
        verbose=False
    ))

    annotated_frame = result.plot()

    boxes = result.boxes

    if boxes.id is not None:
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]

            # סינון לפי גודל בוקס – למניעת false positive
            if w < 30 or h < 30:
                continue

            # אם זה track חדש → עדכן סכום
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                label = model.names[cls_id]
                value = value_map.get(label, 0)
                total_sum += value

            # הדפסת Track ID מעל כל מטבע
            cv2.putText(
                annotated_frame,
                f"ID:{track_id}",
                (int(xyxy[0]), int(xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

    # הצגת סכום בש"ח
    cv2.putText(
        annotated_frame,
        f"Total: {total_sum} \u20AA",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Coin Tracker", annotated_frame)

    # יציאה עם מקש 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
