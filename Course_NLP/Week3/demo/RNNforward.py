#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True) 
        #batch_first=True表示输入数据的形状为[batch_size, seq_len, input_size]
        #在NLP任务中建议加上

    def forward(self, x):
        return self.layer(x)

#自定义RNN模型
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size, sequence_length, input_size = x.shape
        ht = np.zeros((batch_size, self.hidden_size))
        output = []
        for t in range(sequence_length):
            xt = x[:, t, :]
            ux = np.dot(xt, self.w_ih.T)
            wh = np.dot(ht, self.w_hh.T)
            ht_next = np.tanh(ux + wh)
            output.append(ht_next)
            ht = ht_next
        output = np.stack(output, axis=1)
        return output, ht


x = np.array([[[1, 2, 3],
               [3, 4, 5],
               [5, 6, 7]]])  # 网络输入，增加一个维度表示batch_size

#torch实验
hidden_size = 4
torch_model = TorchRNN(3, hidden_size)

# print(torch_model.state_dict())
w_ih = torch_model.state_dict()["layer.weight_ih_l0"].numpy()
w_hh = torch_model.state_dict()["layer.weight_hh_l0"].numpy()
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)
#
torch_x = torch.FloatTensor(x)
output, h = torch_model.forward(torch_x)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("---------------")
diy_model = DiyModel(w_ih, w_hh, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果")
print(h, "diy模型预测隐含层结果")
