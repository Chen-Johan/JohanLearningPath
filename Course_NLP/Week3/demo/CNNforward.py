#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现CNN
手动实现CNN
对比
"""
#一个二维卷积
class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(TorchCNN, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)

    def forward(self, x):
        return self.layer(x)

#自定义CNN模型
class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            kernel_weight = kernel_weight.squeeze().numpy() #shape : 2x2
            kernel_output = np.zeros((self.height - kernel_size + 1, self.width - kernel_size + 1))
            for i in range(self.height - kernel_size + 1):
                for j in range(self.width - kernel_size + 1):
                    window = x[i:i+kernel_size, j:j+kernel_size]
                    kernel_output[i, j] = np.sum(kernel_weight * window) # np.dot(a, b) != a * b
            output.append(kernel_output)
        return np.array(output)


x = np.array([[0.1, 0.2, 0.3, 0.4],
              [-3, -4, -5, -6],
              [5.1, 6.2, 7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])  #网络输入

#torch实验
in_channel = 1
out_channel = 3
kernel_size = 2
torch_model = TorchCNN(in_channel, out_channel, kernel_size)
print(torch_model.state_dict())
torch_w = torch_model.state_dict()["layer.weight"]
# print(torch_w.numpy().shape)
torch_x = torch.FloatTensor([[x]])
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, output.shape, "torch模型预测结果\n")
print("---------------")
diy_model = DiyModel(x.shape[0], x.shape[1], torch_w, kernel_size)
output = diy_model.forward(x)
print(output, "diy模型预测结果")
