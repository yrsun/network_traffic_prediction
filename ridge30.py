from sklearn.linear_model import Ridge
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from collections import Counter
import numpy as np
import pandas as pd


dataframe = read_csv(r"C:\Users\Kuma Baobao\Documents\research\ridge\sd0704_1.csv", usecols=[0], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

features = []
y = []
features_window_size = 30
train_size = 10000


mode_numbers = sorted(Counter(dataset[:train_size, 0]).most_common(8), key=lambda student: student[0])

def adjust(num):
    if(num <= (mode_numbers[0][0]+mode_numbers[1][0]))/2:
        return mode_numbers[0][0]
    n = (mode_numbers[0][0]+mode_numbers[1][0])/2
    for k in range(1,7):
        if(num <= (mode_numbers[k][0]+mode_numbers[k+1][0])/2 and num >= n):
            return mode_numbers[k][0]
        else:
            n = (mode_numbers[k][0]+mode_numbers[k+1][0])/2
    return mode_numbers[7][0]


for k in range(0,train_size):
	temp = dataset[k:(features_window_size + k)].T
	temp = np.array(temp)
	features.append(temp)
	temp2 = dataset[(features_window_size + k):(features_window_size + k + 1)]
	y.append(temp2)

features = np.array(features)
nsamples, nx, ny = features.shape
features = features.reshape((nsamples,nx*ny))

y = np.array(y)
nsamples, nx, ny = y.shape
y = y.reshape((nsamples,nx*ny))

clf = Ridge(alpha=0.01)
clf.fit(features, y)

predict_result = list()
expected_result = list()
for k in range(train_size, len(dataset) - features_window_size):
    temp = dataset[k:(features_window_size + k)].T
    temp = np.array(temp)
    predict = clf.predict(temp)
    predict_result.append(predict[0])
    expected_result.append(dataset[(features_window_size + k)])
    print('predicted=%f, adjust=%f, expected=%f' % (predict[0], adjust(predict[0]), dataset[(features_window_size + k)]))


error = mean_absolute_error(predict_result, expected_result)
error = error * len(predict_result) / sum(expected_result)
print('Test NMAE: %.3f' % error)

