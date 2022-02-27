from __future__ import  print_function
import csv
import numpy as np
from numpy import array
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers import Dropout, Bidirectional
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


'''
train_data_mean = np.zeros(np.shape(train_data)).astype(np.float)

test_data_mean = np.zeros(np.shape(test_data)).astype(np.float)

for a in range(len(train_data)-window_size):
    temp = train_data[a:a+window_size]
    temp_avr = np.average(temp)
    train_data_mean[a:a+window_size] = train_data[a:a+window_size]/temp_avr

for a in range(len(test_data)-window_size):
    temp = test_data[a:a+window_size]
    test_avr = np.average(temp)
    test_data_mean[a:a+window_size] = test_data[a:a+window_size]/temp_avr

#X_train = array([[1,2,30,40],[40,50,40,50], [30,40,50,60] , [40,50,60,70]])
#Y_train = array([50,60,70,80])
X_train = train_data_mean.reshape(96,1,1)
Y_train = test_data_mean.reshape((test_data_mean.shape[0],1,1,1))

print(X_train,X_train.dtype,X_train.shape)
print(Y_train,Y_train.dtype,Y_train.shape)

'''

X, Y = split_sequence(train_data, n_steps)
for i in range(len(X)):
    print(X[i],Y[i])
print(X,X.shape,X.dtype)
print(Y,Y.shape,Y.dtype)

X_input,Y_input = split_sequence(test_data, n_steps)
for i in range(len(X_input)):
    print(X_input[i],Y_input[i])

plt.plot(X, 'ro-')
plt.plot(Y, 'ro-')
n_batch = 1
n_features = 1
X = X.reshape((X.shape[0],X.shape[1], n_features))
 #define model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())
model.add(Bidirectional(LSTM(128,return_sequences=True,activation='relu')))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1,activation='relu'))
#model.add(TimeDistributed(Dense(5,activation='softmax')))
model.compile(optimizer='adam', loss='mse')
print(model.summary())


model.fit(X,Y, epochs=1000,batch_size=n_batch, verbose=0)



X_input = X_input.reshape(X_input.shape[0],X_input.shape[1], n_features)

print('prediction val')
yhat = model.predict(X_input,batch_size=n_batch ,verbose=0)
print(yhat)

'''
model.add(TimeDistributed(Conv1D(filters=1, kernel_size=30, activation='relu'),input_shape=(2976,1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(96, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss ='mse')
print(model.summary())
model.fit(X_train,Y_train, epochs=1000,batch_size=10, verbose=0)
print("training complete")
sore = model.evaluate(X_train, Y_train, verbose= 0)

X_input = X_train
X_input = X_train.reshape((1,1,1,1))
yhat = model.predict(X_input, verbose=0)
print("prediction socre about 50,60,70,80",yhat)
'''
