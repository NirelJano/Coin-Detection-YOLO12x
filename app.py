from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")  # מסלול למודל שלך

# מצלמה — אם 0 לא עובד, נסה 1 או 2
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # זיהוי אובייקטים על הפריים
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()  # מצייר את התוצאות על התמונה

        # שליחה לממשק ב־MJPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
