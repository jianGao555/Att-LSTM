import pandas as pd
import numpy
data_close = pd.read_excel('./shuju/结果.xlsx')["沪深指数"]
#data_close = data_close.astype('float32').values  # 转换数据类型

data_close=numpy.array(data_close)
data_close=data_close.reshape(-1,3)
print(data_close)
pd.DataFrame(data_close.T).to_csv("沪深指数.csv")