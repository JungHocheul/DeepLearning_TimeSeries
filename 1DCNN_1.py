from __future__ import  print_function
import csv
import numpy as np
from numpy import array
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt

def split_sequence(sequence, n_steps):
    X, Y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    return array(X), array(Y)

n_steps = 96

columns_train = defaultdict(list)

with open('HomeF-meter2_2014_mar.csv') as f:
   reader = csv.DictReader(f)
   for row in reader :
      for (k,v) in row.items() :
         columns_train[k].append(v)
train_data = np.array(columns_train['Usage [kW]']).astype(np.float)


columns_test = defaultdict(list)

with open('HomeF-meter2_2014_test.csv') as f:
    reader = csv.DictReader(f)
    for row in reader :
        for (k,v) in row. items() :
            columns_test[k].append(v)
test_data = np.array(columns_test['Usage [kW]']).astype(np.float)


X, Y = split_sequence(train_data, n_steps)
for i in range(len(X)):
    print(X[i],Y[i])
print(X,X.shape,X.dtype)
print(Y,Y.shape,Y.dtype)

X_input,Y_input = split_sequence(test_data, n_steps)
for i in range(len(X_input)):
    print(X_input[i],Y_input[i])

n_features = 1
X = X.reshape((X.shape[0],X.shape[1], n_features))
 #define model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X,Y, epochs=1000, verbose=0)



X_input = X_input.reshape(X_input.shape[0],X_input.shape[1], n_features)

print('prediction val')
yhat = model.predict(X_input, verbose=0)

plt.figure(2)
fig = plt.figure(figsize=(5, 3))
plt.plot(test_data[0:960:1], 'r', label="Validation data")
plt.plot(yhat[0:960:1], 'b', label="Prediction data")
plt.title('4_days Prediction')
plt.xlabel('days')
plt.ylabel('Usage(KW)')
plt.legend(loc='upper left', fontsize='x-large', prop={'size': 10})
plt.show()

