from ultralytics import YOLO
import cv2

# טען את המודל
model = YOLO('best.pt')

# מיפוי תוויות לערכים מספריים
value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10}

# טען תמונה מהדיסק - החלף לנתיב שלך
img_path = 'path/to/your/image.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Error: couldn't load image")
    exit(1)

# הרץ חיזוי
results = model.predict(img)

total = 0

# צייר תיבות וסכום
for r in results:
    boxes = r.boxes
    for box in boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        x1, y1, x2, y2 = xyxy

        cls_id = int(box.cls.cpu().numpy())
        label = model.names.get(cls_id, str(cls_id))

        conf = float(box.conf.cpu().numpy())

        # חישוב סכום לפי תווית
        total += value_map.get(label, 0)

        # ציור תיבה
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf*100:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

print(f"Total sum: {total}")

# הצגת התמונה עם תוצאות
cv2.imshow('Prediction', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
