import csv
import numpy as np
from numpy import genfromtxt

FILENAME='train.csv'

data = genfromtxt(FILENAME, delimiter=',', skip_header=1)

np.random.shuffle(data)
print(data.shape)

#Elevation
data[:,1] /= np.max(data[:,1])
#Aspect and Slope
data[:,[2, 3]] /= 360.0
#Distances
data[:,4] /= np.max(data[:,4])
data[:,5] /= np.max(data[:,5])
data[:,6] /= np.max(data[:,6])
#Hillshades
data[:,[7, 8, 9]] /= 255.0
#Fire Points distance
data[:,10] /= np.max(data[:,10])

colums=np.arange(1, 55)
X = data[:,colums]

Y = data[:,55]
Y -= 1

np.save('X_train', X)
np.save('Y_train', Y)