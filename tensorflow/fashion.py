# Programmer: Adam Fahrer
# Purpose: To build a Model that predict what type of clothing is in a given image 

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = data.load_data()

classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainImages = trainImages / 255.0
testImages = testImages / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainImages,trainLabels,epochs=10)

prediction = model.predict(testImages)
for i in range(0,5):
    plt.grid(False)
    plt.imshow(testImages[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + classNames[testLabels[i]])
    plt.title('Prediction: ' + classNames[np.argmax(prediction[i])])
    plt.show()