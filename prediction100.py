from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error
from numpy.linalg import LinAlgError
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xlrd
import csv

# load data
# df = pd.read_excel('sample_data_0704.xlsx', 'rtps_tgc220_sf1', index_col = 0, parse_cols = "A,B") # the first sheet
df = pd.read_excel('sample_data_0704.xlsx', 'rtps_tgc220_sf1', parse_cols = "A,B") # the first sheet
df = df[:-1] # remove the last empty entry
# df.index = pd.to_datetime(df.index, unit = 'ms')
df = df.dropna()
df['curr_req'] = df['curr_req'].astype(np.float64)
# print(df.values.dtype)
series = df


X = series['curr_req']
y = series['recv time']
size = 100
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

with open('randomWalkResult.csv','w', newline='') as file:
    file = csv.writer(file)
    file.writerow(['time', 'predicted', 'expected'])
    for t in range(len(test)):
        model = ARIMA(history, order=(0,1,0))
        # model = sm.tsa.statespace.SARIMAX(history, trend='n', order=(0, 1, 0), seasonal_order=(0, 2, 1, 12))
        try:
            model_fit = model.fit(method = 'css', disp=0)
        except (ValueError, LinAlgError):
            pass
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t + size]
        history.append(obs)
        history = history[1:]
        print('time=%f, predicted1=%f, expected=%f' % (y[t + size], yhat, obs))
        file.writerow([y[t + size], yhat[0], obs])

print('prediction finished')
predictions = predictions[1:]
test = test[:-1]
error = mean_absolute_error(test, predictions)
error = error * test.size / sum(test)
print('Test NMAE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()