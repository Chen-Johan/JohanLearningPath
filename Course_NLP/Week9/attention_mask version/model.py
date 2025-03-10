# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        max_length = config["max_length"]
        class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):
        x, _ = self.bert(x, attention_mask=attention_mask)
        logits = self.classify(x)
        
        if target is not None:
            if self.use_crf:
                mask = attention_mask.bool() if attention_mask is not None else None
                return -self.crf_layer(logits, target, mask=mask, reduction='mean')
            else:
                return self.loss(logits.view(-1, logits.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                mask = attention_mask.bool() if attention_mask is not None else None
                # 返回解码后的标签序列列表
                return self.crf_layer.decode(logits, mask=mask)
            else:
                # 返回logits tensor
                return logits


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)