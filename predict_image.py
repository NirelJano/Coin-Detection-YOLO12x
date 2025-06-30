from ultralytics import YOLO
import cv2

# טען את המודל
model = YOLO('best.pt')

# מיפוי תוויות לסכומים
value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10,'10Ag':0.1, '50Ag': 0.5,}

# טען את התמונה
img_path = '/Users/shachafemanoel/Documents/Coin-Detection-YOLO12x/WhatsApp Image 2025-06-15 at 12.26.49 (3).jpeg'
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"לא נמצא קובץ תמונה בנתיב: {img_path}")

# הרצת חיזוי
results = model.predict(img, conf=0.5, agnostic_nms=True)[0]

# חילוץ תוצאות
boxes = results.boxes.xyxy.cpu().numpy().astype(int)
classes = results.boxes.cls.cpu().numpy().astype(int)
confs = results.boxes.conf.cpu().numpy()

# סכום כולל
total = 0

for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
    label = model.names.get(cls_id, str(cls_id))
    total += value_map.get(label, 0)

    # ציור תיבה ותיוג
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    text = f"{label} "
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(img, (x1, y1 - th - baseline - 10), (x1 + tw, y1), (0, 0, 0), -1)
    cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

# הצגת הסכום על גבי התמונה
total_text = f"Total: {total} ILS"
(text_width, text_height), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
cv2.rectangle(img, (10, 10), (20 + text_width, 20 + text_height), (0, 0, 0), -1)
cv2.putText(img, total_text, (15, 20 + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

# הצגה
cv2.imshow('Prediction', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
