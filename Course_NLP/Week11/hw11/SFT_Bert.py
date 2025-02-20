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

"""
基于pytorch的Bert语言模型 实现SFT

BERT本身并非生成型模型: BERT是一种双向编码器,
它是为了理解任务而设计的，如句子分类、问答、命名实体识别等。
它并不像GPT那样本身是生成式的,
所以直接用BERT来做生成任务(即根据标题生成内容)并不理想。
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path, tokenizer):
        super(LanguageModel, self).__init__()
        # 加载预训练的BERT模型，设置eager模式提高训练速度
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.tokenizer = tokenizer
        # 输出层，将BERT的hidden_size维转换为词表大小
        self.classify = nn.Linear(hidden_size, vocab_size)
        # 设置ignore_index为[PAD]的token id
        self.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
        # 在计算loss时忽略[PAD]的部分
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,
            # label_smoothing=0.1  # 添加label smoothing防止过拟合
        )

    def forward(self, x, y=None):
        if y is not None:
            # 训练模式：创建梯形注意力掩码
            # 梯形掩码的目的是让模型:
            # 1. content部分可以双向注意（完全可见）
            # 2. title部分只能看到已生成的token（自回归）
            batch_size, seq_len = x.shape[0], x.shape[1]
            
            # 先创建全1矩阵(batch_size, seq_len, seq_len)
            # 初始状态下所有位置都可以相互注意
            mask = torch.ones((batch_size, seq_len, seq_len))
            
            # 对batch中的每个样本分别处理
            for i in range(batch_size):
                sep_pos = (x[i] == self.tokenizer.convert_tokens_to_ids("[SEP]")).nonzero(as_tuple=True)[0][0]
                
                # 修改掩码逻辑 - 现在是从title生成content
                # 1. title部分自回归，content部分能够看到所有内容
                mask[i, :sep_pos, :sep_pos] = torch.tril(torch.ones((sep_pos, sep_pos)))  # title部分自回归
                mask[i, sep_pos:, :sep_pos] = 0  # content部分只能看到title部分
                mask[i, sep_pos:, sep_pos:] = 1  # content部分可以看到自己的上下文

            
            # 如果有GPU，将mask移到GPU上
            if torch.cuda.is_available():
                mask = mask.cuda()
                
            # 将掩码传入BERT，实现注意力控制
            x, _ = self.bert(x, attention_mask=mask)
            # 将BERT的输出映射到词表大小
            y_pred = self.classify(x)
            # 计算损失：将预测值和目标值展平后计算交叉熵
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 推理模式：不需要掩码，因为是一个token一个token地生成
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            # 使用softmax得到概率分布
            return torch.softmax(y_pred, dim=-1)


#加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data)
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


#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, tokenizer, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab_size, char_dim, pretrain_model_path, tokenizer):
    
    model = LanguageModel(char_dim, vocab_size, pretrain_model_path, tokenizer)
    return model


#文本生成测试代码
def generate_sentence(title, model, tokenizer, window_size):
    model.eval()
    with torch.no_grad():
        input_text = title + "[SEP]"
        generated_content = ""
        
        while len(generated_content) <= 150:  # 增加生成内容的长度限制
            current_input = input_text + generated_content
            x = tokenizer.encode(current_input, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            
            y = model(x)[0][-1]  # 只取最后一个时间步的预测
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
            
            if pred_char == "[EOS]":  # 使用[EOS]作为生成终止符
                break
            generated_content += pred_char
            
    return generated_content
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

# def sampling_strategy(prob_distribution):
#     # 随机选择采样策略:90%概率使用贪婪搜索,10%概率使用随机采样
#     if random.random() > 0.1:
#         strategy = "greedy"
#     else:
#         strategy = "sampling"
        
#     if strategy == "greedy":
#         # 贪婪搜索:选择概率最大的token
#         return int(torch.argmax(prob_distribution))
#     elif strategy == "sampling":
#         # 随机采样:根据概率分布随机选择一个token
#         prob_distribution = prob_distribution.cpu().numpy()
#         return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
    
def sampling_strategy(prob_distribution, temperature=0.7):
    # 应用温度缩放
    prob_distribution = prob_distribution / temperature
    
    if random.random() > 0.1:  # 90%使用top-k采样
        # 只保留前k个最高概率
        k = 5
        values, indices = torch.topk(prob_distribution, k)
        prob_distribution = torch.zeros_like(prob_distribution).scatter_(-1, indices, values)
    
    # 使用softmax确保概率和为1
    prob_distribution = torch.softmax(prob_distribution, dim=-1)
    
    # 转换为numpy数组并确保概率和为1
    prob_distribution = prob_distribution.cpu().numpy()
    prob_distribution = prob_distribution / prob_distribution.sum()  # 再次归一化确保和为1
    
    return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

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
    batch_size = 64
    train_sample = 10000
    char_dim = 768
    vocab_size = 21128
    learning_rate = 3e-5
    
    pretrain_model_path = r'C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week6\demo\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)
    
    # 计算最大长度：title + [SEP] + content + [SEP]
    window_size = max([len(data['title']) + 1 + len(data['content']) for data in corpus])
    
    
    # 从tokenizer中获取实际的词表大小
    vocab_size = tokenizer.vocab_size
    
    model = build_model(vocab_size, char_dim, pretrain_model_path, tokenizer)
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
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.5f}")
        
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
        
        # 每轮结束后测试生成效果
        model.eval()
        test_samples = random.sample(corpus, 2)  # 随机抽取2个样本测试
        for sample in test_samples:
            title = sample['title']
            real_content = sample['content']
            generated_content = generate_sentence(title, model, tokenizer, window_size)
            print("\n标题:", title)
            print("真实内容:", real_content[:50] + "...")
            print("生成内容:", generated_content)
            print("-" * 50)
    
    if save_weight:
        os.makedirs("model", exist_ok=True)
        base_name = os.path.basename(corpus_path).replace(".json", ".pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", True)
