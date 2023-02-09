import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from datetime import datetime
today = datetime.today().strftime("%Y%m%d")
hanabank = "086790"
hanadf = fdr.DataReader(hanabank, "20200101", today)
#hanadf = pd.read_csv("hana.csv")

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

hanadfx = hanadf[['Open','High','Low','Volume', 'Close']]
hanadfx_1 = MinMaxScaler(hanadfx)
hanadfy = hanadfx_1[['Close']]
hanadfx = pd.concat([hanadfx_1[['Open','High','Low','Volume']], hanadf[['Change']]], axis = 1, join='inner') #change 추가
print(hanadfx)

X = hanadfx.values.tolist()
y = hanadfy.values.tolist()

window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)
print(len(_X))
print('전체 데이터의 크기 :', len(data_X), len(data_y))

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])
test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])
print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)
print(train_X.shape[1], train_X.shape[2])

from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, SimpleRNN, Dense, Flatten
#modeling
model = Sequential()

#Input Layer
model.add(SimpleRNN(10,
                    activation='relu',
                    input_shape=(train_X.shape[1], train_X.shape[2]),
                    return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#Hidden Layer
model.add(SimpleRNN(50, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(SimpleRNN(25, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(SimpleRNN(15, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#Out Layer
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(train_X, train_y, epochs=100, validation_data=(test_X, test_y), batch_size=20)
pred_y = model.predict(test_X)
print(pred_y[-1])

print("내일 SEC 주가 :", hanadf.iloc[-1][4] * pred_y[-1] / hanadfy.iloc[-1], 'KRW')
print(history.history.keys())

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,1,1)
plt.plot(test_y, color='red', label='real stock price')
plt.plot(pred_y, color='blue', label='predicted stock price')
plt.title('SEC stock price prediction_DNN')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()

plt.subplot(2,1,2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.ylim(0,0.01)
plt.legend()
plt.show()
