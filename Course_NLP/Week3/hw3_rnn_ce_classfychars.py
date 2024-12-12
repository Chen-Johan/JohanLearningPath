#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.rnn = nn.RNN(vector_dim, 128, bias=True, batch_first=True)   #RNN层
        self.fc = nn.Linear(128, len(vocab))  #全连接层
        self.loss = nn.functional.cross_entropy  #损失函数 交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x, _ = self.rnn(x)                           # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, 128)
        x = x[:, -1, :]                              # 取最后一个时间步的输出 (batch_size, sen_len, 128) -> (batch_size, 128)
        y_pred = self.fc(x)                        #(batch_size, 128) -> (batch_size, sen_len + 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #28
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 如果特定字符在字符串中，返回其位置，否则返回-1
def build_sample(vocab, sentence_length, target_char):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if target_char in x:
        y = x.index(target_char)
    else:
        y = sentence_length  # 如果没有特定字符，返回句子长度作为类别
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length, target_char):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, target_char)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, target_char):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, target_char)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(y_p) == int(y_t):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    #配置参数
    target_char = '你'
    epoch_num = 50        #训练轮数
    batch_size = 300       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 8   #样本文本长度
    learning_rate = 0.005 #学习率
    
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, target_char) # 建立1000个用于训练的样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, target_char)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
        
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(char_dim, sentence_length, vocab)     #建立模型
    model.load_state_dict(torch.load(model_path, weights_only=True))  #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        pred_class = torch.argmax(result[i]).item()  # 获取预测的类别
        pred_prob = torch.softmax(result[i], dim=0)[pred_class].item()  # 获取预测的概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, pred_class, pred_prob)) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fnvfedge", "wfgz你dfg", "rqwdfreg", "nkktlwww"]
    predict("model.pth", "vocab.json", test_strings)
