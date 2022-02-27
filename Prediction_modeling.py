# ##################
# ##import library##
# ##################
import csv
from collections import defaultdict
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt

# ######################
# ##define_Function#####
# ######################

# data slicing function #
def create_dataset(signal_data, look_back=1):
    data_x, data_y = [], []
    for i in range(len(signal_data) - look_back):
        data_x.append(signal_data[i:(i + look_back), 0])
        data_y.append(signal_data[i + look_back, 0])
    return np.array(data_x), np.array(data_y)

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

# modeling evalutation function #
# MAE (Mean Squared Error ) under the function is mean_absolute_error( test, pred )
#mean_absolute_error(y_test, y_pred)
# MSE ( Mean Squared Error )
#mean_squared_error(y_test, y_pred)
# RMSE ( Root Mean Squared Error )
# RMSSE Root Mean Squared Error

def root_mean_ssquared_error(y_ture, y_pred, y_test):
    n = len(y_test)
    numerator = np.mean(np.sum(np.square(y_true - y_pred))
    denominator = 1/(n-1)*np.sum(np.square((y_test[1:]- y_test[:-1])))
    msse = numerator/denominator
    return msse**0.5

def root_mean_squared_error(true, test_data, prediction_data):
    MAE = mean_absolute_error(test_data, prediction_data)
    MSE = means_squared_error(test_data, prediction_data)
    RMSSE = root_mean_ssquared_error(true, prediction_data, test_data)
    return MAE, MSE,RMSSE, np.sqrt(MSE)





# ###########################
# ###define global variable##
# ###########################

#look_back = 100
#num_cell = 64
#batch_size = 5
#epoch = 100
#window_size = 48

n_steps = 96


# file name
train_filename = 'HomeF-meter2_2015.csv'
# test_file_name
# val_file_name

# #### Main Process #####
# #######################

# using dataset
columns_train = defaultdict(list)
#columns_val = defaultdict(list)
#columns_test = defaultdict(list)

with open(train_filename) as f:
   reader = csv.DictReader(f)
   for row in reader :
      for (k,v) in row.items() :
         columns_train[k].append(v)
train_data = np.array(columns_train['Usage [kW]']).astype(np.float)

"""
#juseak propcessing

with open('val_data_2weeks_MOD_.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k, v) in row.items():
            columns_val[k].append(v)
val_data = np.array(columns_val['Usage [kW]']).astype(np.float)

with open('test_data_2weeks_MOD_.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k, v) in row.items():
            columns_test[k].append(v)
test_data = np.array(columns_test['Usage [kW]']).astype(np.float)
"""

# ##################### #
# data slicing function #
# ##################### #
X, Y = split_sequence(train_data, n_steps)

# ML ( smoothing )
# Boot strapping
# DL ( Deep Learning )
# modeling evalutate

# save modeling

# plotting graph part #
fig = plt.figure(figsize=(5, 3))
plt.plot(train_data[0:40540:1], 'r', label="Train_data")
plt.title('10_days Prediction')
plt.xlabel('days')
plt.ylabel('Usage(KW)')
plt.legend(loc='upper left', fontsize='x-large', prop={'size': 10})
plt.show()



