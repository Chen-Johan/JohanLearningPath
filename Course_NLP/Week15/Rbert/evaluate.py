# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from sklearn.metrics import classification_report
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.attribute_schema = self.valid_data.dataset.attribute_schema
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"object_acc":0, "attribute_acc": 0, "value_acc": 0, "full_match_acc":0}
        self.model.eval()
        gold = []
        pred = []
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, e1_mask, e2_mask, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            gold += labels.detach().tolist()
            with torch.no_grad():
                batch_pred = self.model(input_id, e1_mask, e2_mask) #不输入labels，使用模型当前参数进行预测
                batch_pred = torch.argmax(batch_pred, dim=-1)
                pred += batch_pred.detach().tolist()
        report = classification_report(np.array(gold), np.array(pred)).rstrip().split("\n")
        self.logger.info(report[0])
        self.logger.info(report[-1])

