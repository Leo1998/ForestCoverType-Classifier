import plaidml.keras
plaidml.keras.install_backend()

import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

X = np.load("X_test.npy")

model = keras.models.load_model('ForestCoverType-1537622945.543794.model')

Y_predict = model.predict(X, batch_size=64)

Y_predict = np.argmax(Y_predict, axis=1)
Y_predict = np.array(Y_predict, dtype=int)
Y_predict += 1


idx = np.arange(15121, 581013)
output = np.column_stack((idx, Y_predict))

np.savetxt('test_prediction.csv', output, fmt='%d', delimiter=',', header="Id,Cover_Type", comments='')
