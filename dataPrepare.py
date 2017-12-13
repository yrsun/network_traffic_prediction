# load and plot dataset
from pandas import read_csv
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from matplotlib import pyplot
import pandas as pd
import numpy as np


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = pd.read_excel('sample_data_0704.xlsx', 'rtps_tgc220_sf1', index_col = 0, parse_cols = "A,B") # the first sheet
series = series[:-1] # remove the last empty entry
series.index = pd.to_datetime(series.index, unit = 'ms')
series = series.dropna()
series['curr_req'] = series['curr_req'].astype(np.float64)
print(series.values.dtype)

# transform to supervised learning
X = series.values
supervised = timeseries_to_supervised(X, 1)
print(supervised.head())

# summarize first few rows
print(series.head())

# line plot
series.plot()
pyplot.show()