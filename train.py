import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

DATASET = "dataset"
IMG_SIZE = 100

labels = os.listdir(DATASET)

X = []
y = []

for label in labels:
    path = os.path.join(DATASET, label)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        X.append(image.flatten())
        y.append(label)

X = np.array(X) / 255.0

print("Training model...")
model = SVC(probability=True)   # VERY IMPORTANT
model.fit(X, y)

joblib.dump(model, "cattle_model.pkl")
joblib.dump(labels, "labels.pkl")

print("MODEL TRAINED & SAVED")