import requests
import json
import finnhub
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
import math

#np.random.seed(0)
X = np.random.uniform(-100,100,size=(10000,))
Y = np.sin(X)

# pre process the data
BATCH_SIZE = 64
TEST_SIZE = len(X) // 10
VALIDATE_SIZE = len(X) // 20

array = np.array(list(zip(X.tolist(),Y.tolist())))

np.random.shuffle(array)
train_date = array[TEST_SIZE:,0]
train_price = array[TEST_SIZE:,1]
test_date = array[:TEST_SIZE,0]
test_price = array[:TEST_SIZE,1]

x_val = train_date[-VALIDATE_SIZE:]
y_val = train_price[-VALIDATE_SIZE:]
train_date = train_date[:-VALIDATE_SIZE]
train_price = train_price[:-VALIDATE_SIZE]

print(f'Lengths, train: {len(train_price)} test {len(test_price)} val {len(x_val)}')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_date,train_price,epochs=3,
          batch_size=BATCH_SIZE,
          validation_data=(x_val,y_val))

model.evaluate(test_date,test_price,batch_size=BATCH_SIZE)

print(model.predict([math.pi,-math.pi,0,math.pi/2,-math.pi/2]))
