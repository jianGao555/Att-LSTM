#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.svm import SVR
def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))

data_close = pd.read_csv('../shuju/1.csv', encoding='gb2312')["收盘价"]
data_close = data_close.astype('float32').values  # 转换数据类型
# 将价格标准化到0~1
max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)
DAYS_FOR_TRAIN = 10
dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

# 划分训练集和测试集，70%作为训练集
train_size = int(len(dataset_x) * 0.9)
train = False
if train == True:
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    model = SVR()  # 随机森林
    model.fit(train_x, train_y)
    joblib.dump(model, "my_random_forest.pkl")

loaded_rf = joblib.load("my_random_forest.pkl")
dataset_x = dataset_x.reshape(-1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
pred_test = loaded_rf.predict(dataset_x[train_size:])
dataset_y=dataset_y[train_size:]
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差

r2 = r2_score(pred_test, dataset_y)
mse = mean_squared_error(pred_test,dataset_y )
mae = mean_absolute_error(pred_test, dataset_y)
count=0
for i in range(len(dataset_y)):
    if i > 0 and dataset_y[i] >= dataset_y[i - 1] and pred_test[i] >= pred_test[i - 1]:
        count += 1
    elif i > 0 and dataset_y[i] < dataset_y[i - 1] and pred_test[i] < pred_test[i - 1]:
        count += 1
print("趋势准确率为：",count/(len(dataset_y)-1))
print('*' * 100)
print("r2", r2)
print("mse:", mse)
print("mae:", mae)
plt.plot(pred_test, 'coral', label='prediction')
plt.plot(dataset_y, 'deepskyblue', label='real')
plt.legend(loc='best')
#plt.show()
plt.close()
