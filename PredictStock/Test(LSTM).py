import pandas as pd
import numpy as np

hanabank = pd.read_csv("hanabank(21.1.27~23.2.3)_utf8.csv")
print(hanabank.head)
print(hanabank.shape)
hanabank = hanabank.sort_values(by='일자')  #일자로 정렬
print(hanabank.iloc[-1][4])

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = hanabank[['시가', '고가', '저가', '종가', '거래량']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['종가']]
dfx = dfx[['시가','고가','저가','거래량']]
#print(dfx)
#print(dfx.describe())

X = dfx.values.tolist()
y = dfy.values.tolist()
print(X)

#10일 기준으로 다음 값 예측
window_size = 10
data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

print('전체 데이터의 크기 :', len(data_X), len(data_y))

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])
test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])
print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
#setting Model
model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(units=1))
model.summary()

#training
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=70, batch_size=30)
pred_y = model.predict(test_X)

#print MSE, RMSE, R2
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(test_y, pred_y)
RMSE = np.sqrt(MSE)
R2 = r2_score(test_y, pred_y)
print(MSE, RMSE, R2)


#to Graph
import matplotlib.pyplot as plt
plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print(dfy.iloc[-1])
print("내일 SEC 주가 :", hanabank.iloc[-1][4] * pred_y[-1] / dfy.iloc[-1], 'KRW')
