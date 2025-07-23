from ultralytics import YOLO
import cv2

# טען את המודל שלך
model = YOLO("best.pt")

# ערכים כספיים לפי תוויות
value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10, '10Ag': 0.1, '50Ag': 0.5}

# מעקב אחר מטבעות שנספרו כדי למנוע ספירה כפולה
counted_coins = set()

# פתיחת מצלמה
cap = cv2.VideoCapture(1)  # 0 עבור מצלמה ראשית, 1 עבור מצלמה שנייה וכו'
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
        conf=0.3,  # הורדת סף ביטחון לשיפור זיהוי
        verbose=False,
        vid_stride=1  # הפחתת vid_stride לשיפור רגישות
    ))

    annotated_frame = result.plot()

    boxes = result.boxes
    current_frame_coins = set()

    if boxes.id is not None:
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].item())
            cls_id = int(boxes.cls[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]

            # סינון לפי גודל בוקס – למניעת false positive (הפחתת הסף)
            if w < 20 or h < 20:
                continue

            # הוספת מטבע לסט הנוכחי
            current_frame_coins.add(track_id)

            # אם זה מטבע חדש, הוסף אותו לסכום הכולל
            if track_id not in counted_coins:
                label = model.names.get(cls_id, "Unknown")
                value = value_map.get(label, 0)
                counted_coins.add(track_id)

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

    # הסרת מטבעות שלא נראים יותר
    counted_coins &= current_frame_coins

    # חישוב הסכום הכולל
    total_sum = 0
    if boxes.id is not None:
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].item())
            if track_id in counted_coins:
                cls_id = int(boxes.cls[i].item())
                label = model.names.get(cls_id, "Unknown")
                value = value_map.get(label, 0)
                total_sum += value

    # הצגת סכום בש"ח
    cv2.putText(
        annotated_frame,
        f"Total: {total_sum:.2f} ILS",
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
