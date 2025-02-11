import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""

import csv

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {1: '好评', 0: '差评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)

        # 若模型类型为 bert，则初始化 BERT 分词器；否则使用普通词表
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    # def load(self):
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             tag = line["tag"]
    #             label = self.label_to_index[tag]
    #             title = line["title"]
    #             if self.config["model_type"] == "bert":
    #                 input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
    #             else:
    #                 input_id = self.encode_sentence(title)
    #             input_id = torch.LongTensor(input_id)
    #             label_index = torch.LongTensor([label])
    #             self.data.append([input_id, label_index])
    #     return
    
    def load(self):
        # 使用 CSV 方式读取文件，跳过表头
        self.data = []
        with open(self.path, newline='', encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过 CSV 表头
            for row in reader:
                # row[0]为标签，row[1]为评价文本
                label_str, review = row[0], row[1]
                label = int(label_str)

                # 如果是 BERT 模型，使用 self.tokenizer；否则用 encode_sentence() 编码
                if self.config["model_type"] == "bert":
                    encoded_inputs = self.tokenizer.encode_plus(
                        review,
                        max_length=self.config["max_length"],
                        padding='max_length',  # 填充到最大长度
                        truncation=True,       # 截断
                        return_tensors='pt',   # 返回 PyTorch 张量
                        return_attention_mask=True  # 返回 attention mask
                    )
                    input_id = encoded_inputs['input_ids'].squeeze(0)  # 去掉 batch 维度
                    attention_mask = encoded_inputs['attention_mask'].squeeze(0)  # 去掉 batch 维度
                else:
                    input_id = self.encode_sentence(review)
                    attention_mask = torch.ones(self.config["max_length"], dtype=torch.long)  # 对于非 BERT 模型，attention mask 全为 1
                
                
                # if self.config["model_type"] == "bert":
                #     input_id = self.tokenizer.encode(
                #         review,
                #         max_length=self.config["max_length"],
                #         pad_to_max_length=True
                #     )
                # else:
                #     input_id = self.encode_sentence(review)

                # # 转换为 PyTorch 张量后保存到 self.data
                # input_id = torch.LongTensor(input_id)
                # label_index = torch.LongTensor([label])
                # self.data.append([input_id, label_index])
                
                # 转换为 PyTorch 张量后保存到 self.data
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, attention_mask, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
    dg = DataGenerator("train.csv", Config)
    print(dg[1])
