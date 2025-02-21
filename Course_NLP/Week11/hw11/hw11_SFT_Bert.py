#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
sft的一种实现, 主要通过mask和loss计算来实现, Ber作为backbone

BERT本身并非生成型模型: BERT是一种双向编码器,
它是为了理解任务而设计的，如句子分类、问答、命名实体识别等。

"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # 加载预训练的BERT模型，设置eager模式提高训练速度
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')

        # 输出层，将BERT的hidden_size维转换为词表大小
        self.classify = nn.Linear(hidden_size, vocab_size)

        # 在计算loss时忽略[PAD]的部分
        self.loss = nn.CrossEntropyLoss(ignore_index=-1,
            # label_smoothing=0.1  # 添加label smoothing防止过拟合
        )

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            print(mask.shape)
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料, 用title当成假想的prompt，content当成假想的answer
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus

'''
语料数量:  104
最长content长度:  109
最长content+title长度:  135
最长title长度:  28
content长度均值:  105.05
title长度均值:  19.61
'''

#随机生成一个样本
def build_sample(tokenizer, window_size, corpus):
    # 随机选择一条数据
    data = random.choice(corpus)
    
    # 构建输入序列：title + [SEP] + content
    # [SEP]用于分隔title和content
    input_text = data['title'] + "[SEP]" + data['content']
    
    # 找到[SEP]的位置，用于确定content开始生成的位置
    sep_pos = len(data['title']) + 1
    
    # 构建目标序列：
    # 1. title部分用[PAD]填充，因为不需要预测
    # 2. content部分保持原文，这是我们要预测的
    # 3. 最后加[EOS]表示生成结束
    target_text = "[PAD]" * sep_pos + data['content'] + "[EOS]"
    
    # 将文本转换为token ids，设置最大长度和截断
    x = tokenizer.encode(input_text, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)
    y = tokenizer.encode(target_text, add_special_tokens=False, padding='max_length', truncation=True, max_length=window_size)

    return x, y
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
# def build_sample(tokenizer, window_size, corpus):
#     start = random.randint(0, len(corpus) - 1 - window_size)
#     end = start + window_size
    
#     window = corpus[start:end]
#     target = corpus[start + 1:end + 1]  #输入输出错开一位

#     x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)   #将字转换成序号
#     y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=10)

#     return x, y


#sft的数据构造
#loss只计算答案按部分，通过mask矩阵，让上下文之间没有交互
#label中使用-1，表示不参与训练
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        #构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
        mask = create_mask(len(prompt_encode), len(answer_encode))
        #padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#构造掩码，输入两个字符串的长度
def create_mask(s1, s2):
    len_s1 = s1 + 2 #cls + sep
    len_s2 = s2 + 1 #sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0  
    # 遍历s2的每个token
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask

def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

# 修改 build_model 函数调用
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)


# def generate_sentence(openings, model, tokenizer, window_size):
#     # reverse_vocab = dict((y, x) for x, y in vocab.items())
#     model.eval()
#     with torch.no_grad():
#         pred_char = ""
#         #生成了换行符，或生成文本超过30字则终止迭代
#         while pred_char != "\n" and len(openings) <= 30:
#             openings += pred_char
#             x = tokenizer.encode(openings, add_special_tokens=False)
#             x = torch.LongTensor([x])
#             if torch.cuda.is_available():
#                 x = x.cuda()
#             y = model(x)[0][-1]
#             index = sampling_strategy(y)
#             pred_char = ''.join(tokenizer.decode(index))
#     return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
    
# def sampling_strategy(prob_distribution, temperature=0.7):
#     # 应用温度缩放
#     prob_distribution = prob_distribution / temperature
    
#     if random.random() > 0.1:  # 90%使用top-k采样
#         # 只保留前k个最高概率
#         k = 5
#         values, indices = torch.topk(prob_distribution, k)
#         prob_distribution = torch.zeros_like(prob_distribution).scatter_(-1, indices, values)
    
#     # 使用softmax确保概率和为1
#     prob_distribution = torch.softmax(prob_distribution, dim=-1)
    
#     # 转换为numpy数组进行采样
#     prob_distribution = prob_distribution.cpu().numpy()
#     return np.random.choice(len(prob_distribution), p=prob_distribution)

# def sampling_strategy(prob_distribution, temperature=0.7):
#     # 应用温度控制生成的多样性
#     prob_distribution = prob_distribution / temperature
    
#     # 使用top-p (nucleus) sampling
#     sorted_probs, sorted_indices = torch.sort(prob_distribution, descending=True)
#     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
#     # 只保留累积概率达到p的tokens
#     p = 0.9
#     sorted_indices_to_remove = cumulative_probs > p
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     sorted_indices_to_remove[..., 0] = 0
    
#     # 将不需要的概率置为0
#     indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
#     prob_distribution = prob_distribution.masked_fill(indices_to_remove, 0.0)
    
#     # 重新归一化概率分布
#     prob_distribution = prob_distribution / prob_distribution.sum()
    
#     # 转换为numpy进行采样
#     prob_distribution = prob_distribution.cpu().numpy()
#     return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)



def train(corpus_path, save_weight=True):
    epoch_num = 100
    batch_size = 128
    char_dim = 768
    max_length = 50
    vocab_size = 21128
    learning_rate = 3e-4
    
    pretrain_model_path = r'C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week6\demo\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  #建立数据集
    
    
    model = build_model(vocab_size, char_dim, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("模型加载完毕，开始训练")
    best_loss = float('inf')
    no_improve = 0
    patience = 3  # 早停的耐心值
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        
        # 增加验证集评估
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            # 保存最佳模型
            if save_weight:
                os.makedirs("model", exist_ok=True)
                model_path = os.path.join("model", "best_model.pth")
                torch.save(model.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break
        
        # 修改这部分代码
        model.eval()
        test_samples = random.sample(corpus, 2)  # 随机抽取2个样本测试
        for title, content in test_samples:  # 直接解包列表
            print("\n标题:", title)
            print("真实内容:", content)
            generated_content = generate_sentence(title, model, tokenizer)
            print("生成内容:", generated_content)
            print("-" * 50)
    
    if save_weight:
        os.makedirs("model", exist_ok=True)
        base_name = os.path.basename(corpus_path).replace(".json", ".pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    train("sample_data.json", False)
