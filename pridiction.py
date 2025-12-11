import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

perc = load_model("models/perc.h5")
ann = load_model("models/ann.h5")
cnn = load_model("models/cnn.h5")

def preprocess_custom_image(path):
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28, 28))

    img = 255 - img

    img = img.astype("float32") / 255.0

    img_flat = img.reshape(1, 28, 28)

    img_cnn = img.reshape(1, 28, 28, 1)

    return img, img_flat, img_cnn


def predict_custom(path):
    img, img_flat, img_cnn = preprocess_custom_image(path)

    p1 = np.argmax(perc.predict(img_flat))
    p2 = np.argmax(ann.predict(img_flat))
    p3 = np.argmax(cnn.predict(img_cnn))

    plt.imshow(img, cmap="gray")
    plt.title(f"PERC: {p1} | ANN: {p2} | CNN: {p3}")
    plt.axis("off")
    plt.show()

predict_custom("example.jpg")
