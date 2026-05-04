import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

data = []
labels = []

# dataset path
dataset_path = "dataset"

# load images
for category in ["with_mask", "without_mask"]:
    path = os.path.join(dataset_path, category)
    label = 0 if category == "with_mask" else 1

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        data.append(image)
        labels.append(label)

# normalize data
data = np.array(data, dtype="float32") / 255.0
labels = to_categorical(labels, num_classes=2)

# split data
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# load MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

# build head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# compile model
model.compile(loss="binary_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])

# train model
print("Training started...")
model.fit(trainX, trainY,
          validation_data=(testX, testY),
          epochs=10,
          batch_size=32)

# save model
model.save("mask_model.h5")

print("Training completed. Model saved as mask_model.h5")