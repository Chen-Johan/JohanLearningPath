# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample_triplet() #随机生成一个训练样本
        else:
            return self.data[index]

    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    # def random_train_sample(self):
    #     standard_question_index = list(self.knwb.keys())
    #     #随机正样本
    #     if random.random() <= self.config["positive_sample_rate"]:
    #         p = random.choice(standard_question_index)
    #         #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
    #         if len(self.knwb[p]) < 2:
    #             return self.random_train_sample()
    #         else:
    #             s1, s2 = random.sample(self.knwb[p], 2)
    #             return [s1, s2, torch.LongTensor([1])]
    #     #随机负样本
    #     else:
    #         p, n = random.sample(standard_question_index, 2)
    #         s1 = random.choice(self.knwb[p])
    #         s2 = random.choice(self.knwb[n])
    #         return [s1, s2, torch.LongTensor([-1])]
        
    #随机生成三元组样本，两个正样本，一个负样本
    def random_train_sample_triplet(self):
        standard_question_index = list(self.knwb.keys())  # 获取所有标准问题的索引列表
        # 选两个正样本，一个负样本
        while True:
            p = random.choice(standard_question_index)  # 随机选择一个标准问题索引作为正样本
            if len(self.knwb[p]) >= 2:  # 确保正样本对应的问题列表中至少有两个问题
                break
        n = random.choice(standard_question_index)  # 随机选择一个标准问题索引作为负样本
        while n == p:  # 确保负样本和正样本不同
            n = random.choice(standard_question_index)
        s1, s2 = random.sample(self.knwb[p], 2)  # 从正样本对应的问题列表中随机选择两个不同的问题
        s3 = random.choice(self.knwb[n])  # 从负样本对应的问题列表中随机选择一个问题
        
        # 前两个样本是正样本，最后一个样本是负样本，不需要额外输入一个0/1的标签，这与一般的loss计算不同
        return [s1, s2, s3]  # 返回两个正样本和一个负样本组成的三元组


#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
