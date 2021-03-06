import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # list of given number of points per each class
    y = np.zeros(points*classes, dtype='uint8')  # same as above
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))  # index in class
        X[ix] = np.c_[np.random.randn(points)*.1 + class_number/3, np.random.randn(points)*.1 + 0.5]
    y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
