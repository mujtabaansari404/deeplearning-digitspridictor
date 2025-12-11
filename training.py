
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron    

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential     

from tensorflow.keras.layers import Dense     
from tensorflow.keras.layers import Conv2D     
from tensorflow.keras.layers import Flatten    

from tensorflow.keras.layers import MaxPooling2D     
from tensorflow.keras.layers import Dropout          

from tensorflow.keras.utils import to_categorical 

df = pd.read_csv("archive\mnist_train.csv")
df_test = pd.read_csv("archive\mnist_test.csv")
print(df.head())

X_train = df.drop("label", axis=1).values
y_train = df["label"].values
X_test = df_test.drop("label", axis=1).values
y_test = df_test["label"].values
     

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
     

X_train_img = X_train.reshape(-1, 28, 28)
X_test_img = X_test.reshape(-1, 28, 28)
     

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
     

perceptron = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(10, activation="softmax")
])
     

perceptron.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
     

history_percp = perceptron.fit(X_train_img, y_train_cat, epochs=10, batch_size=32,
validation_data=(X_test_img, y_test_cat), verbose=1)

acc_percp = perceptron.evaluate(X_test_img, y_test_cat, verbose=1)[1]
     

print(f"percp accuracy = {acc_percp}")

ann = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_ann = ann.fit(X_train_img, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test_img, y_test_cat), verbose=1)

acc_ann = ann.evaluate(X_test_img, y_test_cat, verbose=0)[1]
print(f"ann accuracy = {acc_ann}!")

X_train_cnn = X_train.reshape(-1, 28, 28,1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
     

cnn = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])


cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
     

history_cnn = cnn.fit(X_train_cnn, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_cnn, y_test_cat), verbose=1)

acc_cnn = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
     

print(f"cnn accuracy = {acc_cnn}")

perceptron_save = perceptron.save("perc.h5")
ann = ann.save("ann.h5")
cnn = cnn.save("cnn.h5")


     

