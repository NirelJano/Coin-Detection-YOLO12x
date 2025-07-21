from flask import Flask, render_template_string, Response, request, jsonify, send_file
from live_web import process_frame
import io
import base64
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="he">
<head>
    <meta charset="UTF-8">
    <title>זיהוי מטבעות בלייב</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            background: #111;
            color: #fff;
        }
        #container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            min-height: 100vh;
        }
        #processed {
            width: 98vw;
            max-width: 700px;
            height: 70vw;
            max-height: 80vh;
            background: #222;
            border-radius: 12px;
            margin-top: 2vh;
            object-fit: contain;
            display: none;
        }
        #controls {
            margin: 2vh 0;
            display: flex;
            gap: 1em;
        }
        button {
            font-size: 1.1em;
            padding: 0.7em 1.5em;
            border-radius: 8px;
            border: none;
            background: #2a9d8f;
            color: #fff;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:active {
            background: #21867a;
        }
        #video {
            width: 1px;
            height: 1px;
            opacity: 0;
            position: absolute;
            left: -9999px;
            top: -9999px;
        }
        #canvas { display: none !important; }
        #spinner {
            display: none;
            margin-top: 3vh;
            border: 8px solid #eee;
            border-top: 8px solid #2a9d8f;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #error-msg {
            color: #ff5555;
            margin-top: 2vh;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>זיהוי מטבעות בלייב</h1>
        <div id="controls">
            <button id="toggle-detect">הפעל זיהוי בלייב</button>
            <button id="switch-camera">החלף מצלמה</button>
        </div>
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas"></canvas>
        <img id="processed"/>
        <div id="spinner"></div>
        <div id="error-msg"></div>
    </div>
    <script>
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let processed = document.getElementById('processed');
        let detectBtn = document.getElementById('toggle-detect');
        let switchBtn = document.getElementById('switch-camera');
        let spinner = document.getElementById('spinner');
        let errorMsg = document.getElementById('error-msg');
        let stream = null;
        let useBackCamera = true;
        let detecting = false;
        let detectTimeout = null;
        let sending = false;

        async function startCamera() {
            errorMsg.textContent = '';
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            let constraints = {
                video: {
                    facingMode: useBackCamera ? { exact: "environment" } : "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                await video.play();
            } catch (e) {
                errorMsg.textContent = 'לא ניתן להפעיל מצלמה: ' + e.message;
                stopDetection();
            }
        }

        switchBtn.onclick = () => {
            useBackCamera = !useBackCamera;
            if (detecting) startCamera();
        };

        detectBtn.onclick = () => {
            detecting = !detecting;
            if (detecting) {
                detectBtn.textContent = 'כבה זיהוי בלייב';
                processed.style.display = '';
                spinner.style.display = '';
                startCamera().then(startDetection);
            } else {
                detectBtn.textContent = 'הפעל זיהוי בלייב';
                processed.style.display = 'none';
                spinner.style.display = 'none';
                stopDetection();
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
            }
        };

        function startDetection() {
            processed.style.display = '';
            spinner.style.display = '';
            sending = false;
            detectLoop();
        }
        function stopDetection() {
            clearTimeout(detectTimeout);
            sending = false;
        }
        async function detectLoop() {
            if (!detecting) return;
            if (sending) {
                detectTimeout = setTimeout(detectLoop, 100);
                return;
            }
            sending = true;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            let ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            let dataUrl = canvas.toDataURL('image/jpeg');
            try {
                let res = await fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });
                if (res.ok) {
                    let blob = await res.blob();
                    processed.src = URL.createObjectURL(blob);
                    spinner.style.display = 'none';
                } else {
                    spinner.style.display = 'none';
                }
            } catch (e) {
                spinner.style.display = 'none';
                errorMsg.textContent = 'שגיאה בשליחת תמונה לשרת';
            }
            sending = false;
            detectTimeout = setTimeout(detectLoop, 500);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        # המרת base64 ל-numpy array
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)
        # עיבוד התמונה
        processed_np = process_frame(img_np)
        # המרת numpy ל-JPEG
        _, buffer = cv2.imencode('.jpg', processed_np)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')
    except Exception as e:
        print("ERROR in /process_frame:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 