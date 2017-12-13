from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error

import pandas as pd

# load data
df = pd.read_excel('video_4K.xlsx', '40mbps_h264', index_col = 0, parse_cols = "A,B") # the first sheet

df = df[:-1] # remove the last empty entry
df.index = pd.to_datetime(df.index, unit = 'ms')
print(df.head())
series = df

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

X = series.values
size = int(len(X) * 0.85)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_absolute_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()