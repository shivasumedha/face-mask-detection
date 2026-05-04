from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load face detector only (TensorFlow removed)
face_net = cv2.dnn.readNet(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# -------- DETECT FUNCTION --------
def detect_mask(frame):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        if detections[0,0,i,2] > 0.5:
            return "Face Detected 🙂"

    return "No Face ❌"

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

# -------- SAFE START --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)