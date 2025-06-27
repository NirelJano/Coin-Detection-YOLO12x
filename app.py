from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO('/Users/innadaymand/Desktop/Coin Detection YOLO12x/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    header, encoded = data.split(',', 1)
    img_bytes = base64.b64decode(encoded)

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)

    boxes_data = []
    total = 0

    class_names = model.names  # שמות הקלאסים במודל
    value_map = {'One': 1, 'Two': 2, 'Five': 5, 'Ten': 10}  # מיפוי ערכים

    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            x1, y1, x2, y2 = xyxy
            label = class_names.get(cls_id, str(cls_id))

            boxes_data.append({
                'label': label,
                'confidence': conf,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2
            })

            total += value_map.get(label, 0)

    return jsonify({'total': total, 'boxes': boxes_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
