import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model


def create_dataset(signal_data, look_back=1):
    data_x, data_y = [], []
    for i in range(len(signal_data) - look_back):
        data_x.append(signal_data[i:(i + look_back), 0])
        data_y.append(signal_data[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


look_back = 100
num_cell = 64
batch_size = 5
epoch = 100
window_size = 48

columns_train = defaultdict(list)

with open('train_data_2months_MOD.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k, v) in row.items():
            columns_train[k].append(v)

# print(columns['sum'])
train_data = np.array(columns_train['sum']).astype(np.float)

columns_val = defaultdict(list)

with open('val_data_2weeks_MOD_.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k, v) in row.items():
            columns_val[k].append(v)

# print(columns['sum'])
val_data = np.array(columns_val['sum']).astype(np.float)

columns_test = defaultdict(list)

with open('test_data_2weeks_MOD_.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k, v) in row.items():
            columns_test[k].append(v)

# print(columns['sum'])
test_data = np.array(columns_test['sum']).astype(np.float)

train_data_mean = np.zeros(np.shape(train_data)).astype(np.float)
val_data_mean = np.zeros(np.shape(val_data)).astype(np.float)
# test_data_mean = np.zeros(np.shape(test_data)).astype(np.float)

for a in range(len(train_data) - window_size):
    temp = train_data[a:a+window_size]
    temp_avg = np.average(temp)
    train_data_mean[a:a+window_size] = train_data[a:a+window_size] / temp_avg

for a in range(len(val_data) - window_size):
    temp = val_data[a:a+window_size]
    temp_avg = np.average(temp)
    val_data_mean[a:a+window_size] = val_data[a:a+window_size] / temp_avg
"""
for a in range(len(test_data) - window_size):
    temp = test_data[a:a+window_size]
    temp_avg = np.average(temp)
    test_data_mean[a:a+window_size] = test_data[a:a+window_size] / temp_avg
"""
train_data_mean = train_data_mean.transpose()
val_data_mean = val_data_mean.transpose()
test_data = test_data.transpose()

"""
print(np.shape(train_data))
print(np.shape(val_data))
print(np.shape(test_data))
"""
train_data_mean = train_data_mean.reshape(2880, 1)
val_data_mean = val_data_mean.reshape(670, 1)
test_data = test_data.reshape(816, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_mean = scaler.fit_transform(train_data_mean)
val_data_mean = scaler.fit_transform(val_data_mean)
test_data = scaler.fit_transform(test_data)

x_train, y_train = create_dataset(train_data_mean, look_back)
x_val, y_val = create_dataset(val_data_mean, look_back)
x_test, y_test = create_dataset(test_data, look_back)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(num_cell, input_shape=(None, 1)))
model.add(Dropout(0.3))
# model.add(LSTM(num_cell, input_shape=(None, 1), return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(num_cell))
# model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

hist = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_val, y_val))

model.save('LSTM_test_ext.h5')

model = load_model('LSTM_test_ext.h5')

look_ahead = len(test_data) - look_back
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))

for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=batch_size)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
plt.legend()
plt.show()

