import matplotlib.pyplot as plt
import numpy as np
losses=np.loadtxt("log_att.txt")
plt.plot(losses, 'coral', label='Shanghai and Shenzhen index prediction')
# plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
plt.legend(loc='best')
#plt.grid(axis="y")
plt.ylabel("")
# plt.savefig('result_attention_Value.png', format='png')
plt.show()