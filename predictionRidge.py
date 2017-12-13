from sklearn.linear_model import Ridge
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

# load data
df = pd.read_excel('sample_data_0704.xlsx', 'rtps_tgc220_sf1', parse_cols = "A,B") # the first sheet
df = df[:-1] # remove the last empty entry
df = df.dropna()
df['curr_req'] = df['curr_req'].astype(np.float64)
series = df


y = series['curr_req']
X = series['recv time']
size = 15
totallen = 60
trainx, testx = X[0:size], X[size:totallen]
trainy, testy = y[0:size], y[size:totallen]
historyx = [x for x in trainx]
historyy = [x for x in trainy]
predictions = list()
print(len(testx))
print(len(y))


for t in range(len(testx)):
    # print(test.iloc[[2]]['recv time'])
    ridgereg = Ridge(alpha=1e-15, normalize=True)
    ridgereg.fit(np.array([historyx]).T, np.array([historyy]).T)
    output = ridgereg.predict(np.array([testx]).T)
    yhat = output[0]
    predictions.append(yhat)
    historyy.append(testy[t+size])
    historyx.append(testx[t+size])
    historyy = historyy[1:]
    historyx = historyx[1:]
    print('predicted=%f, expected=%f' % (yhat, testy[t+size]))


print('prediction finished')
error = mean_absolute_error(testy, predictions)
error = error * len(predictions) / sum(testy)
print('Test NMAE: %.3f' % error)

pyplot.plot(testx, testy)
pyplot.plot(testx, predictions, color='red')
pyplot.show()