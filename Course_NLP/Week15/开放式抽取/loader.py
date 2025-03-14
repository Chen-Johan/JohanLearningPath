# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = {"B_object":0,
                       "I_object":1,
                       "B_attribute":2,
                       "I_attribute":3,
                       "B_value":4,
                       "I_value":5,
                       "O":6}
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]
        self.neglect_count = 0
        self.load()
        print("超出设定最大长度的样本数量:%d, 占比:%.3f"%(self.exceed_max_length, self.exceed_max_length/len(self.data)))
        print("由于属性不存在于原文中，忽略%d条样本, 占比:%.3f"%(self.neglect_count, self.neglect_count / len(self.data)))

    def load(self):
        self.text_data = []
        self.data = []
        self.exceed_max_length = 0
        with open(self.path, encoding="utf8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                object = sample["object"]
                attribute = sample["attribute"]
                value = sample["value"]
                # 对于基于序列标注的开放式抽取，如果目标属性（或关系）不出现在原文，则无法抽取
                if attribute not in context:
                    self.neglect_count += 1
                    continue
                self.text_data.append([context, object, attribute, value]) #在测试时使用
                input_id, label = self.process_sentence(context, object, attribute, value)
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
        return

    def process_sentence(self, context, object, attribute, value):
        if len(context) > self.max_length:
            self.exceed_max_length += 1
        object_start = context.index(object)
        attribute_start = context.index(attribute)
        value_start = context.index(value)
        input_id = self.encode_sentence(context)
        assert len(context) == len(input_id)
        label = [self.schema["O"]] * len(input_id)
        #标记实体
        label[object_start] = self.schema["B_object"]
        for index in range(object_start + 1, object_start + len(object)):
            label[index] = self.schema["I_object"]
        # 标记属性
        label[attribute_start] = self.schema["B_attribute"]
        for index in range(attribute_start + 1, attribute_start + len(attribute)):
            label[index] = self.schema["I_attribute"]
        # 标记属性值
        label[value_start] = self.schema["B_value"]
        for index in range(value_start + 1, value_start + len(value)):
            label[index] = self.schema["I_value"]

        input_id = self.padding(input_id, 0)
        label = self.padding(label, -100)
        return input_id, label

    def encode_sentence(self, text, padding=False):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

