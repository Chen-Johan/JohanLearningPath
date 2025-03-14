# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())
        config["num_labels"] = len(self.attribute_schema)
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def process_sentence(self, context, enti1, enti2):
        enti1_start = context.index(enti1) + 1  #因为bert的分词会在第一位增加[cls]，所以向后移动一位
        enti2_start = context.index(enti2) + 1    #同上

        input_id = self.tokenizer.encode(context)
        # 标记头实体
        e1_mask = [0] * len(input_id)
        for index in range(enti1_start, enti1_start + len(enti1)):
            e1_mask[index] = 1
        # 标记尾实体
        e2_mask = [0] * len(input_id)
        for index in range(enti2_start, enti2_start + len(enti2)):
            e2_mask[index] = 1

        return input_id, e1_mask, e2_mask

    def predict(self, sentence, enti1, enti2):
        input_id, e1_mask, e2_mask = self.process_sentence(sentence, enti1, enti2)
        with torch.no_grad():
            relation_pred = self.model(torch.LongTensor([input_id]),
                                       torch.LongTensor([e1_mask]),
                                       torch.LongTensor([e2_mask])
                                       )
            relation_pred = torch.argmax(relation_pred)
            relation = self.index_to_label[int(relation_pred)]
        return relation

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_10.pth")

    sentence = "可你知道吗，兰博基尼的命名取自创始人“费鲁吉欧·兰博基尼”的名字，而更让人意外的是，兰博基尼刚开始只是一个做拖拉机的！"
    e1 = "兰博基尼"
    e2 = "费鲁吉欧·兰博基尼"
    res = sl.predict(sentence, e1, e2)
    print("预测关系：", res)

    sentence = "傻丫头郭芙蓉、大女人翠平、励志的杜拉拉，姚晨的角色跳跃很大，是一个颇能适应各种类型题材的职业演员。"
    e1 = "姚晨"
    e2 = "演员"
    res = sl.predict(sentence, e1, e2)
    print("预测关系：", res)

