from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
face_net = cv2.dnn.readNet("models/deploy.prototxt",
                           "models/res10_300x300_ssd_iter_140000.caffemodel")
mask_model = load_model("mask_model.h5")

camera = cv2.VideoCapture(0)
last_frame = None

# -------- DETECT FUNCTION --------
def detect_mask(frame):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    label = "No Face"

    for i in range(detections.shape[2]):
        if detections[0,0,i,2] > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224,224)) / 255.0
            face = np.reshape(face, (1,224,224,3))

            mask, noMask = mask_model.predict(face, verbose=0)[0]

            return "Mask 😷" if mask > noMask else "No Mask ❌"

    return label

# -------- LIVE --------
def generate_frames():
    global last_frame

    while True:
        success, frame = camera.read()
        if not success:
            break

        last_frame = frame.copy()
        label = detect_mask(frame)

        color = (0,255,0) if "Mask" in label else (0,0,255)
        cv2.putText(frame, label, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------- CAPTURE --------
@app.route("/capture")
def capture():
    global last_frame

    if last_frame is None:
        return jsonify({"result":"Camera not ready"})

    return jsonify({"result": detect_mask(last_frame)})

# -------- MAIN --------
@app.route("/", methods=["GET","POST"])
def index():
    result = None
    image_file = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            img = cv2.imread(path)
            result = detect_mask(img)
            image_file = "uploads/" + file.filename

    return render_template("index.html", result=result, image=image_file)

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)