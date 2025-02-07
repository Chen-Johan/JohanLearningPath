# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.max_length = config["max_length"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"],
                                                       #use_fast=True,
                                                       add_special_tokens=True)
        #{'I-Reply', 'I-Review', 'B-Reply', 'B-Review', 'O'}
        self.label_map = {"B":0, "I":1, "O":2}
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for segment in f.read().split("\n\n"):
                if segment.strip() == "" or "\n" not in segment:
                    continue
                self.prepare_data(segment)
        return


    def prepare_data(self, segment):
        segment_input_ids = []
        segment_attention_mask = []
        labels = []  
        for line in segment.split("\n"):
            if line.strip() == "":
                continue
            sentence = line.split("\t")[0]
            label = line.split("\t")[1][0]
            role = line.split("\t")[3]
            assert role in ["Reply", "Review"]
            assert label in self.label_map, label
            label = self.label_map[label]
            encode = self.tokenizer.encode_plus(sentence,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                add_special_tokens=True)
            input_ids = encode["input_ids"]
            attention_mask = encode["attention_mask"]
            #token_type_ids = encode["token_type_ids"]
            segment_input_ids.append(input_ids)
            segment_attention_mask.append(attention_mask)
            #segment_token_type_ids.append(token_type_ids)
            labels.append(label)
            if len(labels) > self.config["max_sentence"]:
                break
        self.data.append([torch.LongTensor(segment_input_ids),
                          torch.LongTensor(segment_attention_mask),
                          #torch.LongTensor(segment_token_type_ids),
                          torch.LongTensor(labels)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    # dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dg


if __name__ == "__main__":
    from config import Config
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


