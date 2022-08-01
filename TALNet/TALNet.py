#!/usr/bin/python3
# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import time

DAYS_FOR_TRAIN = 5

import math


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归

        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.Transformer_model = nn.TransformerEncoderLayer(d_model=5, nhead=1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        _x = self.Transformer_model(_x)
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集

        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


if __name__ == '__main__':
    t0 = time.time()
    data_close = pd.read_csv('../shuju/1.csv', encoding='gb2312')["收盘价"]
    data_close = data_close.astype('float32').values  # 转换数据类型

    # 将价格标准化到0~1
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

    # 划分训练集和测试集，70%作为训练集
    train_size = int(len(dataset_x) * 0.9)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)

    # 转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    model = LSTM_Regression(DAYS_FOR_TRAIN, 32, output_size=1, num_layers=1)  # 导入模型并设置模型的参数输入输出层、隐藏层等

    model_total = sum([param.nelement() for param in model.parameters()])  # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total / 1e6))

    train_loss = []
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for i in range(200):
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())

        # 将训练过程的损失值写入文档保存，并在终端打印出来
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i + 1, loss.item()))
        if (i + 1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i + 1, loss.item()))

    # 画loss曲线
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()

    torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
    t1 = time.time()
    T = t1 - t0
    print('The training time took %.2f' % (T / 60) + ' mins.')

    tt0 = time.asctime(time.localtime(t0))
    tt1 = time.asctime(time.localtime(t1))
    print('The starting time was ', tt0)
    print('The finishing time was ', tt1)

    # for test
    model = model.eval()  # 转换成测试模式
    model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

    # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)

    pred_test = model(dataset_x[train_size:])  # 测试训练集
    dataset_y = dataset_y[train_size:]
    # pred_test = model(dataset_x)  # 全量训练集

    # 的模型输出 (seq_size, batch_size, output_size)
    pred_test = pred_test.view(-1).data.numpy()
    # pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
    # assert len(pred_test) == len(data_close)
    # pred_test=pred_test*(max_value - min_value)+min_value
    # dataset_y=dataset_y*(max_value - min_value)+min_value
    # pd.DataFrame(pred_test).to_csv("pre_USD.csv")

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error  # 均方误差
    from sklearn.metrics import mean_absolute_error  # 平方绝对误差

    r2 = r2_score(pred_test, dataset_y)
    mse = mean_squared_error(pred_test, dataset_y)
    mae = mean_absolute_error(pred_test, dataset_y)
    count = 0
    for i in range(len(dataset_y)):
        if i > 0 and dataset_y[i] >= dataset_y[i - 1] and pred_test[i] >= pred_test[i - 1]:
            count += 1
        elif i > 0 and dataset_y[i] < dataset_y[i - 1] and pred_test[i] < pred_test[i - 1]:
            count += 1
    print("趋势准确率为：", count / (len(dataset_y) - 1))
    print('*' * 100)
    print("r2", r2)
    print("mse:", mse)
    print("mae:", mae)
    plt.plot(pred_test, 'coral', label='prediction')
    plt.plot(dataset_y, 'deepskyblue', label='real')
    # plt.plot((lenth, lenth), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.show()
    plt.close()

