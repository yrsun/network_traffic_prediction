import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense

def readData(dataset_tgc220_64, dataset_tgc220_128, dataset_arris_64, dataset_arris_128, dataset_video4k):
    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_10MBPS = data['10mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_20MBPS = data['20mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_30MBPS = data['30mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_40MBPS = data['40mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_50MBPS = data['50mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_60MBPS = data['60mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_70MBPS = data['70mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_80MBPS = data['80mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_90MBPS = data['90mbps']

    data = pd.read_csv('tgc220_64.csv')
    data = data.dropna()
    tgc220_64_100MBPS = data['100mbps']

    tgc220_64_10MBPS = tgc220_64_10MBPS.astype(np.float64)
    tgc220_64_20MBPS = tgc220_64_20MBPS.astype(np.float64)
    tgc220_64_30MBPS = tgc220_64_30MBPS.astype(np.float64)
    tgc220_64_40MBPS = tgc220_64_40MBPS.astype(np.float64)
    tgc220_64_50MBPS = tgc220_64_50MBPS.astype(np.float64)
    tgc220_64_60MBPS = tgc220_64_60MBPS.astype(np.float64)
    tgc220_64_70MBPS = tgc220_64_70MBPS.astype(np.float64)
    tgc220_64_80MBPS = tgc220_64_80MBPS.astype(np.float64)
    tgc220_64_90MBPS = tgc220_64_90MBPS.astype(np.float64)
    tgc220_64_100MBPS = tgc220_64_100MBPS.astype(np.float64)

    dataset_tgc220_64.append(tgc220_64_10MBPS)
    dataset_tgc220_64.append(tgc220_64_20MBPS)
    dataset_tgc220_64.append(tgc220_64_30MBPS)
    dataset_tgc220_64.append(tgc220_64_40MBPS)
    dataset_tgc220_64.append(tgc220_64_50MBPS)
    dataset_tgc220_64.append(tgc220_64_60MBPS)
    dataset_tgc220_64.append(tgc220_64_70MBPS)
    dataset_tgc220_64.append(tgc220_64_80MBPS)
    dataset_tgc220_64.append(tgc220_64_90MBPS)
    dataset_tgc220_64.append(tgc220_64_100MBPS)


    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_10MBPS = data['10mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_20MBPS = data['20mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_30MBPS = data['30mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_40MBPS = data['40mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_50MBPS = data['50mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_60MBPS = data['60mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_70MBPS = data['70mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_80MBPS = data['80mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_90MBPS = data['90mbps']

    data = pd.read_csv('arris_64.csv')
    data = data.dropna()
    arris_64_100MBPS = data['100mbps']

    arris_64_10MBPS = arris_64_10MBPS.astype(np.float64)
    arris_64_20MBPS = arris_64_20MBPS.astype(np.float64)
    arris_64_30MBPS = arris_64_30MBPS.astype(np.float64)
    arris_64_40MBPS = arris_64_40MBPS.astype(np.float64)
    arris_64_50MBPS = arris_64_50MBPS.astype(np.float64)
    arris_64_60MBPS = arris_64_60MBPS.astype(np.float64)
    arris_64_70MBPS = arris_64_70MBPS.astype(np.float64)
    arris_64_80MBPS = arris_64_80MBPS.astype(np.float64)
    arris_64_90MBPS = arris_64_90MBPS.astype(np.float64)
    arris_64_100MBPS = arris_64_100MBPS.astype(np.float64)

    dataset_arris_64.append(arris_64_10MBPS)
    dataset_arris_64.append(arris_64_20MBPS)
    dataset_arris_64.append(arris_64_30MBPS)
    dataset_arris_64.append(arris_64_40MBPS)
    dataset_arris_64.append(arris_64_50MBPS)
    dataset_arris_64.append(arris_64_60MBPS)
    dataset_arris_64.append(arris_64_70MBPS)
    dataset_arris_64.append(arris_64_80MBPS)
    dataset_arris_64.append(arris_64_90MBPS)
    dataset_arris_64.append(arris_64_100MBPS)


    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_10MBPS = data['10mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_20MBPS = data['20mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_40MBPS = data['40mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_60MBPS = data['60mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_80MBPS = data['80mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_90MBPS = data['90mbps']

    data = pd.read_csv('tgc220_128.csv')
    data = data.dropna()
    tgc220_128_100MBPS = data['100mbps']

    tgc220_128_10MBPS = tgc220_128_10MBPS.astype(np.float64)
    tgc220_128_20MBPS = tgc220_128_20MBPS.astype(np.float64)
    tgc220_128_40MBPS = tgc220_128_40MBPS.astype(np.float64)
    tgc220_128_60MBPS = tgc220_128_60MBPS.astype(np.float64)
    tgc220_128_80MBPS = tgc220_128_80MBPS.astype(np.float64)
    tgc220_128_90MBPS = tgc220_128_90MBPS.astype(np.float64)
    tgc220_128_100MBPS = tgc220_128_100MBPS.astype(np.float64)

    dataset_tgc220_128.append(tgc220_128_10MBPS)
    dataset_tgc220_128.append(tgc220_128_20MBPS)
    dataset_tgc220_128.append(tgc220_128_40MBPS)
    dataset_tgc220_128.append(tgc220_128_60MBPS)
    dataset_tgc220_128.append(tgc220_128_80MBPS)
    dataset_tgc220_128.append(tgc220_128_90MBPS)
    dataset_tgc220_128.append(tgc220_128_100MBPS)


    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_10MBPS = data['10mbps']

    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_20MBPS = data['20mbps']

    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_40MBPS = data['40mbps']

    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_60MBPS = data['60mbps']

    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_80MBPS = data['80mbps']

    data = pd.read_csv('arris_128.csv')
    data = data.dropna()
    arris_128_90MBPS = data['90mbps']

    arris_128_10MBPS = arris_128_10MBPS.astype(np.float64)
    arris_128_20MBPS = arris_128_20MBPS.astype(np.float64)
    arris_128_40MBPS = arris_128_40MBPS.astype(np.float64)
    arris_128_60MBPS = arris_128_60MBPS.astype(np.float64)
    arris_128_80MBPS = arris_128_80MBPS.astype(np.float64)
    arris_128_90MBPS = arris_128_90MBPS.astype(np.float64)

    dataset_arris_128.append(arris_128_10MBPS)
    dataset_arris_128.append(arris_128_20MBPS)
    dataset_arris_128.append(arris_128_40MBPS)
    dataset_arris_128.append(arris_128_60MBPS)
    dataset_arris_128.append(arris_128_80MBPS)
    dataset_arris_128.append(arris_128_90MBPS)

    data = pd.read_csv('video4k.csv')
    data = data.dropna()
    video4k = data['req size']

    video4k = video4k.astype(np.float64)

    dataset_video4k.append(video4k)

def processOneSheet(train_data, test_data, sheet, c):
    for dataset in sheet:
        # create training and testing sample index
        total_index = list(range(0, dataset.size))
        train_index = random.sample(total_index, int(len(total_index) * 0.7))
        test_index = list(set(total_index) - set(train_index))
        count = 500
        for i in train_index:
            count -= 1
            if (count == 0):
                break
            if i < len(total_index) - window_size - 1:
                t = list(dataset[i: i + window_size])
                t.append(c)
                train_data.append(t)
        for i in test_index:
            count -= 1
            if (count == 0):
                break
            if i < len(total_index) - window_size - 1:
                t = list(dataset[i: i + window_size])
                t.append(c)
                test_data.append(t)

def processData(train_data, test_data, dataset_tgc220_64, dataset_tgc220_128, dataset_arris_64, dataset_arris_128, video4k):
    processOneSheet(train_data, test_data, dataset_tgc220_64[:int(np.shape(dataset_tgc220_64)[0] / 2)], 0)
    processOneSheet(train_data, test_data, dataset_tgc220_64[int(np.shape(dataset_tgc220_64)[0] / 2):], 1)
    processOneSheet(train_data, test_data, dataset_tgc220_128[:int(np.shape(dataset_tgc220_128)[0] / 2)], 0)
    processOneSheet(train_data, test_data, dataset_tgc220_128[int(np.shape(dataset_tgc220_128)[0] / 2):], 1)
    processOneSheet(train_data, test_data, dataset_arris_64[:int(np.shape(dataset_arris_64)[0] / 2)], 2)
    processOneSheet(train_data, test_data, dataset_arris_64[int(np.shape(dataset_arris_64)[0] / 2):], 3)
    processOneSheet(train_data, test_data, dataset_arris_128[:int(np.shape(dataset_arris_128)[0] / 2)], 2)
    processOneSheet(train_data, test_data, dataset_arris_128[int(np.shape(dataset_arris_128)[0] / 2):], 3)
    processOneSheet(train_data, test_data, video4k, 4)
    # processOneSheet(train_data, test_data, dataset_arris_64, 2)
    # processOneSheet(train_data, test_data, dataset_arris_128, 3)

window_size = 10

dataset_tgc220_64 = []  # (10, 10469)
dataset_tgc220_128 = [] # (7, 7561)
dataset_arris_64 = []   # (10, 10618)
dataset_arris_128 = []  # (6, 11174)
video4k = []


readData(dataset_tgc220_64, dataset_tgc220_128, dataset_arris_64, dataset_arris_128, video4k)
print(np.shape(video4k))
train_data = []
test_data = []

processData(train_data, test_data, dataset_tgc220_64, dataset_tgc220_128, dataset_arris_64, dataset_arris_128, video4k)
random.shuffle(train_data)
train_data = np.array(train_data)
test_data = np.array(test_data)



X = train_data[:,0:window_size]
Y = train_data[:,window_size]

print(np.shape(X))
print(np.shape(Y))

# create model
model = Sequential()
model.add(Dense(window_size, input_dim=window_size, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=30, batch_size=100,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]

X = test_data[:,0:window_size]
Y = test_data[:,window_size]
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




