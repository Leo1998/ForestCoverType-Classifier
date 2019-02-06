import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

X = np.load("X_train.npy")

Y = np.load("Y_train.npy")
Y = keras.utils.to_categorical(Y, 7)


model = Sequential()

model.add(Dense(128, activation="relu", input_shape=X.shape[1:]))
model.add(Dropout(0.1))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(7, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y, epochs=20, batch_size=64, validation_split=0.0)

model.save("ForestCoverType-{}.model".format(time.time()))