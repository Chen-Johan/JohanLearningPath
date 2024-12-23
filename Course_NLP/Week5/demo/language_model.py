#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
基于pytorch的语言模型
与基于窗口的词向量训练本质上非常接近
只是输入输出的预期不同
不使用向量的加和平均，而是直接拼接起来
"""

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len, embedding_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.word_vectors = nn.Embedding(vocab_size, embedding_size)
        self.inner_projection_layer = nn.Linear(embedding_size * max_len, hidden_size)
        self.outter_projection_layer = nn.Linear(hidden_size, hidden_size)
        self.x_projection_layer = nn.Linear(embedding_size * max_len, hidden_size)
        self.projection_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, context):
        #context shape = batch_size, max_length
        context_embedding = self.word_vectors(context)  #output shape = batch_size, max_length, embedding_size
        #总体计算 y = b+Wx+Utanh(d+Hx)， 其中x为每个词向量的拼接
        #词向量的拼接
        x = context_embedding.view(context_embedding.shape[0], -1) #shape = batch_size, max_length*embedding_size
        #hx + d
        inner_projection = self.inner_projection_layer(x)  #shape = batch_size, hidden_size
        #tanh(hx+d)
        inner_projection = torch.tanh(inner_projection)    #shape = batch_size, hidden_size
        #U * tanh(hx+d) + b
        outter_project = self.outter_projection_layer(inner_projection)  # shape = batch_size, hidden_size
        #Wx
        x_projection = self.x_projection_layer(x)    #shape = batch_size, hidden_size
        #y = Wx + Utanh(hx+d) + b
        y = x_projection + outter_project  #shape = batch_size, hidden_size
        #softmax后输出预测概率, 训练的目标是让y_pred对应到字表中某个字
        y_pred = torch.softmax(y, dim=-1)  #shape = batch_size, hidden_size
        return y_pred

vocab_size = 8  #词表大小
embedding_size = 5  #人为指定的向量维度
max_len = 4 #输入长度
hidden_size = vocab_size  #由于最终的输出维度应当是字表大小的，所以这里hidden_size = vocab_size
model = LanguageModel(vocab_size, max_len, embedding_size, hidden_size)
#假如选取一个文本窗口“天王盖地虎”
#输入：“天王盖地” —> 输出："虎"
#假设词表embedding中, 天王盖地虎 对应位置 12345
context = torch.LongTensor([[1,2,3,4]])  #shape = 1, 4  batch_size = 1, max_length = 4
pred = model(context)
print("预测值：", pred)
print("loss可以使用交叉熵计算：", nn.functional.cross_entropy(pred, torch.LongTensor([5])))


print("词向量矩阵")
matrix = model.state_dict()["word_vectors.weight"]

print(matrix.shape)  #vocab_size, embedding_size
print(matrix)