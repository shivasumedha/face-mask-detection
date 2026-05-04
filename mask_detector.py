import cv2
import numpy as np
import time
import pygame
import os
from tensorflow.keras.models import load_model

# 🔊 sound init
pygame.mixer.init()
alert_sound = "alert.mp3"
ok_sound = "alright.mp3"

# 📁 ensure folder exists
os.makedirs("captured_images", exist_ok=True)

# load models
prototxt_path = "models/deploy.prototxt"
weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNet(prototxt_path, weights_path)
model = load_model("mask_model.h5")

cap = cv2.VideoCapture(0)

last_alert_time = 0
last_ok_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    mask_found = False
    no_mask_found = False

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))

            (mask, noMask) = model.predict(face, verbose=0)[0]

            label = "Mask" if mask > noMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            text = f"{label}: {max(mask, noMask)*100:.2f}%"

            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), color, 2)

            if label == "Mask":
                mask_found = True
            else:
                no_mask_found = True

    current_time = time.time()

    # 🔴 No Mask → alert + capture
    if no_mask_found:
        if current_time - last_alert_time > 3:
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play()

            filename = f"captured_images/no_mask_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

            last_alert_time = current_time

    # 🟢 All Mask → ok sound
    elif mask_found:
        if current_time - last_ok_time > 5:
            pygame.mixer.music.load(ok_sound)
            pygame.mixer.music.play()
            last_ok_time = current_time

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()