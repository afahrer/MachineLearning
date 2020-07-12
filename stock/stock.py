import requests
import json
import finnhub
import time
import tensorflow as tf
import numpy as np
from datetime import datetime

print("Stock Name")
stock = input()

print("Date DD-MM-YYYY")
predict_date = input() + ' 12:00:00'
predict_date = datetime.strptime(predict_date, "%d-%m-%Y %H:%M:%S")
predict_date = time.mktime(predict_date.timetuple())
# Configure API key
configuration = finnhub.Configuration(
    api_key={
        'token': 'bs528mvrh5rb6mgkdcu0'
    })

MIN_DATE = 946684800.0
MAX_DATE = 1893456000.0
finnhub_client = finnhub.DefaultApi(finnhub.ApiClient(configuration))
prices = finnhub_client.stock_candles(stock, '1', int(MIN_DATE) , int(time.time()))

# pre process the data
BATCH_SIZE = 64
TEST_SIZE = len(prices.o) // 10
VALIDATE_SIZE = len(prices.o) // 20

times = np.array(prices.t)
predict_date = np.array([predict_date])
predict_date = (predict_date - MIN_DATE) / MAX_DATE - MIN_DATE
times = (times - MIN_DATE) / MAX_DATE - MIN_DATE
array = np.array(list(zip(prices.o,times.tolist())))

np.random.shuffle(array)
train_date = array[TEST_SIZE:,1]
train_price = array[TEST_SIZE:,0]
test_date = array[:TEST_SIZE,1]
test_price = array[:TEST_SIZE,0]

x_val = train_date[-VALIDATE_SIZE:]
y_val = train_price[-VALIDATE_SIZE:]
train_date = train_date[:-VALIDATE_SIZE]
train_price = train_price[:-VALIDATE_SIZE]

print(f'Lengths, train: {len(train_price)} test {len(test_price)} val {len(x_val)}')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), name='date'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_date,train_price,epochs=5,
          batch_size=BATCH_SIZE,
          validation_data=(x_val,y_val))

model.evaluate(test_date,test_price,batch_size=BATCH_SIZE)

print(model.predict(predict_date))
