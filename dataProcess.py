# preprocess all datasets

import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import csv

datasets = []

# 30 mbps.csv

data2 = pd.read_csv('30mbps.csv')
data2 = data2.drop('upstream avg rate', 1)
data2 = data2.dropna()
reqSize2 = data2['req size']

# 40 mbps.csv
data3 = pd.read_csv('40mbps.csv')
data3 = data3.drop('upstream avg rate', 1)
data3 = data3.dropna()
data3 = data3.astype(float)
reqSize3 = data3['req size']

# 20 mbps.csv
data1 = pd.read_csv('h265.csv')
data1 = data1.drop('upstream avg rate', 1)
data1 = data1.dropna()
reqSize1 = data1['req size']

data4 = pd.read_csv('july4_rtps_arris.csv')
data4 = data4.drop('interval', 1)
data4 = data4.dropna()
reqSize6 = data4['curr_req']

data5 = pd.read_csv('80mbps.csv')
data5 = data5.drop('upstream average rate', 1)
data5 = data5.dropna()
reqSize5 = data5['request size']

data6 = pd.read_csv('80mbps_thom.csv')
data6 = data6.drop('upstream average rate', 1)
data6 = data6.dropna()
reqSize4 = data6['request size']

data7 = pd.read_csv('jul4_thom.csv')
data7 = data7.dropna()
reqSize7 = data7['curr_req']

data8 = pd.read_csv('jul4_tgc220.csv')
data8 = data8.dropna()
reqSize8 = data8['curr_req']

reqSize4 = reqSize4.astype(np.float64)
reqSize5 = reqSize5.astype(np.float64)
reqSize6 = reqSize6.astype(np.float64)
reqSize7 = reqSize7.astype(np.float64)
reqSize8 = reqSize8.astype(np.float64)

datasets.append(reqSize1)
datasets.append(reqSize2)
datasets.append(reqSize3)
datasets.append(reqSize4)
datasets.append(reqSize5)
datasets.append(reqSize6)
datasets.append(reqSize7)
datasets.append(reqSize8)

data1 = pd.read_csv('tgc220_64.csv')
data1 = data1.dropna()
tgc220_tenMBPS = data1['10mbps']

data2 = pd.read_csv('tgc220_64.csv')
data2 = data2.dropna()
tgc220_twentyMPBS = data2['20mbps']

data3 = pd.read_csv('tgc220_64.csv')
data3 = data3.dropna()
tgc220_thirtyMPBS = data3['30mbps']

data4 = pd.read_csv('tgc220_64.csv')
data4 = data4.dropna()
tgc220_fourtyMBPS = data4['40mbps']

data5 = pd.read_csv('tgc220_64.csv')
data5 = data5.dropna()
tgc220_fiftyMBPS = data5['50mbps']

data6 = pd.read_csv('tgc220_64.csv')
data6 = data6.dropna()
tgc220_sixtyMBPS = data6['60mbps']

data7 = pd.read_csv('tgc220_64.csv')
data7 = data7.dropna()
tgc220_seventyMBPS = data7['70mbps']

data8 = pd.read_csv('tgc220_64.csv')
data8 = data8.dropna()
tgc220_eightyMBPS = data8['80mbps']

data9 = pd.read_csv('tgc220_64.csv')
data9 = data9.dropna()
tgc220_ninetyMBPS = data9['90mbps']

data10 = pd.read_csv('tgc220_64.csv')
data10 = data10.dropna()
tgc220_hundredMBPS = data10['100mbps']

tgc220_tenMBPS = tgc220_tenMBPS.astype(np.float64)
tgc220_twentyMBPS = tgc220_twentyMPBS.astype(np.float64)
tgc220_thirtyMBPS = tgc220_thirtyMPBS.astype(np.float64)
tgc220_fourtyMBPS = tgc220_fourtyMBPS.astype(np.float64)
tgc220_fiftyMBPS = tgc220_fiftyMBPS.astype(np.float64)
tgc220_sixtyMBPS = tgc220_sixtyMBPS.astype(np.float64)
tgc220_seventyMBPS = tgc220_seventyMBPS.astype(np.float64)
tgc220_eightyMBPS = tgc220_eightyMBPS.astype(np.float64)
tgc220_ninetyMBPS = tgc220_ninetyMBPS.astype(np.float64)
tgc220_hundredMBPS = tgc220_hundredMBPS.astype(np.float64)

datasets.append(tgc220_tenMBPS)
datasets.append(tgc220_twentyMBPS)
datasets.append(tgc220_thirtyMBPS)
datasets.append(tgc220_fourtyMBPS)
datasets.append(tgc220_fiftyMBPS)
datasets.append(tgc220_sixtyMBPS)
datasets.append(tgc220_seventyMBPS)
datasets.append(tgc220_eightyMBPS)
datasets.append(tgc220_ninetyMBPS)
datasets.append(tgc220_hundredMBPS)

data1 = pd.read_csv('arris_64.csv')
data1 = data1.dropna()
arris64_tenMBPS = data1['10mbps']

data2 = pd.read_csv('arris_64.csv')
data2 = data2.dropna()
arris64_twentyMPBS = data2['20mbps']

data3 = pd.read_csv('arris_64.csv')
data3 = data3.dropna()
arris64_thirtyMPBS = data3['30mbps']

data4 = pd.read_csv('arris_64.csv')
data4 = data4.dropna()
arris64_fourtyMBPS = data4['40mbps']

data5 = pd.read_csv('arris_64.csv')
data5 = data5.dropna()
arris64_fiftyMBPS = data5['50mbps']

data6 = pd.read_csv('arris_64.csv')
data6 = data6.dropna()
arris64_sixtyMBPS = data6['60mbps']

data7 = pd.read_csv('arris_64.csv')
data7 = data7.dropna()
arris64_seventyMBPS = data7['70mbps']

data8 = pd.read_csv('arris_64.csv')
data8 = data8.dropna()
arris64_eightyMBPS = data8['80mbps']

data9 = pd.read_csv('arris_64.csv')
data9 = data9.dropna()
arris64_ninetyMBPS = data9['90mbps']

data10 = pd.read_csv('arris_64.csv')
data10 = data10.dropna()
arris64_hundredMBPS = data10['100mbps']

arris64_tenMBPS = arris64_tenMBPS.astype(np.float64)
arris64_twentyMBPS = arris64_twentyMPBS.astype(np.float64)
arris64_thirtyMBPS = arris64_thirtyMPBS.astype(np.float64)
arris64_fourtyMBPS = arris64_fourtyMBPS.astype(np.float64)
arris64_fiftyMBPS = arris64_fiftyMBPS.astype(np.float64)
arris64_sixtyMBPS = arris64_sixtyMBPS.astype(np.float64)
arris64_seventyMBPS = arris64_seventyMBPS.astype(np.float64)
arris64_eightyMBPS = arris64_eightyMBPS.astype(np.float64)
arris64_ninetyMBPS = arris64_ninetyMBPS.astype(np.float64)
arris64_hundredMBPS = arris64_hundredMBPS.astype(np.float64)

datasets.append(arris64_tenMBPS)
datasets.append(arris64_twentyMBPS)
datasets.append(arris64_thirtyMBPS)
datasets.append(arris64_fourtyMBPS)
datasets.append(arris64_fiftyMBPS)
datasets.append(arris64_sixtyMBPS)
datasets.append(arris64_seventyMBPS)
datasets.append(arris64_eightyMBPS)
datasets.append(arris64_ninetyMBPS)
datasets.append(arris64_hundredMBPS)

data1 = pd.read_csv('tgc220_128.csv')
data1 = data1.dropna()
tgc220_128_tenMBPS = data1['10mbps']

data2 = pd.read_csv('tgc220_128.csv')
data2 = data2.dropna()
tgc220_128_twentyMPBS = data2['20mbps']

data4 = pd.read_csv('tgc220_128.csv')
data4 = data4.dropna()
tgc220_128_fourtyMBPS = data4['40mbps']

data6 = pd.read_csv('tgc220_128.csv')
data6 = data6.dropna()
tgc220_128_sixtyMBPS = data6['60mbps']

data8 = pd.read_csv('tgc220_128.csv')
data8 = data8.dropna()
tgc220_128_eightyMBPS = data8['80mbps']

data9 = pd.read_csv('tgc220_128.csv')
data9 = data9.dropna()
tgc220_128_ninetyMBPS = data9['90mbps']

data10 = pd.read_csv('tgc220_128.csv')
data10 = data10.dropna()
tgc220_128_hundredMBPS = data10['100mbps']

tgc220_128_tenMBPS = tgc220_128_tenMBPS.astype(np.float64)
tgc220_128_twentyMBPS = tgc220_128_twentyMPBS.astype(np.float64)
tgc220_128_fourtyMBPS = tgc220_128_fourtyMBPS.astype(np.float64)
tgc220_128_sixtyMBPS = tgc220_128_sixtyMBPS.astype(np.float64)
tgc220_128_eightyMBPS = tgc220_128_eightyMBPS.astype(np.float64)
tgc220_128_ninetyMBPS = tgc220_128_ninetyMBPS.astype(np.float64)
tgc220_128_hundredMBPS = tgc220_128_hundredMBPS.astype(np.float64)

datasets.append(tgc220_128_tenMBPS)
datasets.append(tgc220_128_twentyMBPS)
datasets.append(tgc220_128_fourtyMBPS)
datasets.append(tgc220_128_sixtyMBPS)
datasets.append(tgc220_128_eightyMBPS)
datasets.append(tgc220_128_ninetyMBPS)
datasets.append(tgc220_128_hundredMBPS)

data1 = pd.read_csv('arris_128.csv')
arris128_tenMBPS = data1['10mbps']
arris128_tenMBPS = arris128_tenMBPS.dropna()

data2 = pd.read_csv('arris_128.csv')
arris128_twentyMBPS = data2['20mbps']
arris128_twentyMBPS = arris128_twentyMBPS.dropna()

data4 = pd.read_csv('arris_128.csv')
arris128_fourtyMBPS = data4['40mbps']
arris128_fourtyMBPS = arris128_fourtyMBPS.dropna()

data6 = pd.read_csv('arris_128.csv')
arris128_sixtyMBPS = data6['60mbps']
arris128_sixtyMBPS = arris128_sixtyMBPS.dropna()

data8 = pd.read_csv('arris_128.csv')
arris128_eightyMBPS = data8['80mbps']
arris128_eightyMBPS = arris128_eightyMBPS.dropna()

data9 = pd.read_csv('arris_128.csv')
arris128_ninetyMBPS = data9['90mbps']
arris128_ninetyMBPS = arris128_ninetyMBPS.dropna()

arris128_tenMBPS = arris128_tenMBPS.astype(np.float64)
arris128_twentyMBPS = arris128_twentyMBPS.astype(np.float64)
arris128_fourtyMBPS = arris128_fourtyMBPS.astype(np.float64)
arris128_sixtyMBPS = arris128_sixtyMBPS.astype(np.float64)
arris128_eightyMBPS = arris128_eightyMBPS.astype(np.float64)
arris128_ninetyMBPS = arris128_ninetyMBPS.astype(np.float64)

datasets.append(arris128_tenMBPS)
datasets.append(arris128_twentyMBPS)
datasets.append(arris128_fourtyMBPS)
datasets.append(arris128_sixtyMBPS)
datasets.append(arris128_eightyMBPS)
datasets.append(arris128_ninetyMBPS)