# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel

"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = {"B_object":0,
                       "I_object":1,
                       "B_attribute":2,
                       "I_attribute":3,
                       "B_value":4,
                       "I_value":5,
                       "O":6}
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")


    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def decode(self, pred_label, context):
        pred_label = "".join([str(i) for i in pred_label.detach().tolist()])
        pred_obj = self.seek_pattern("01*", pred_label, context)
        pred_attribute = self.seek_pattern("23*", pred_label, context)
        pred_value = self.seek_pattern("45*", pred_label, context)
        return pred_obj, pred_attribute, pred_value

    def seek_pattern(self, pattern, pred_label, context):
        pred_obj = re.search(pattern, pred_label)
        if pred_obj:
            s, e = pred_obj.span()
            pred_obj = context[s:e]
        else:
            pred_obj = ""
        return pred_obj

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res = torch.argmax(res, dim=-1)
        object, attribute, value = self.decode(res, sentence)
        return object, attribute, value

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_15.pth")

    sentence = "曾任中华民国总统的蒋中正、蒋经国、李登辉皆信仰基督教；"
    res = sl.predict(sentence)
    print(res)

    sentence = "马拉博是赤道几内亚首都，位于比奥科岛北端。"
    res = sl.predict(sentence)
    print(res)
