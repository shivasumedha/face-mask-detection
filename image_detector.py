import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# 🔊 sound init
pygame.mixer.init()
alert_sound = "alert.mp3"
ok_sound = "alright.mp3"

# load models
prototxt_path = "models/deploy.prototxt"
weights_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNet(prototxt_path, weights_path)
model = load_model("mask_model.h5")

# load image
image_path = "test.jpg"
frame = cv2.imread(image_path)

(h, w) = frame.shape[:2]

blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

alert_triggered = False
ok_triggered = False

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

        if label == "No Mask" and not alert_triggered:
            pygame.mixer.music.load(alert_sound)
            pygame.mixer.music.play()
            alert_triggered = True

        elif label == "Mask" and not ok_triggered:
            pygame.mixer.music.load(ok_sound)
            pygame.mixer.music.play()
            ok_triggered = True

cv2.imshow("Image Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()