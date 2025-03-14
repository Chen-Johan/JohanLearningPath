# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
复现论文Relation Bert
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.hidden_size = self.bert.config.hidden_size
        self.cls_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.entity_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_labels = self.config["num_labels"]
        self.label_classifier = nn.Linear(self.hidden_size * 3, self.num_labels)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(0.5)

    #entity mask 形如：    [0,0,1,1,0,0,..]
    def entity_average(self, hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, sentence_length]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, sentence_length] * [b, sentence_length, hidden_size]
        # = [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # 除以实体词长度，求平均
        return avg_vector

    def forward(self, input_ids, e1_mask, e2_mask, labels=None):
        outputs = self.bert(input_ids)
        sequence_output = outputs[0] # batch, sen_len, hidden_size
        pooled_output = outputs[1]  # [CLS]   batch, hidden_size

        # 实体向量求平均
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        # dropout
        e1_h = self.dropout(e1_h)
        e2_h = self.dropout(e2_h)
        pooled_output = self.dropout(pooled_output)

        #过线性层并激活
        pooled_output = self.activation(self.cls_fc_layer(pooled_output))
        e1_h = self.activation(self.entity_fc_layer(e1_h))
        e2_h = self.activation(self.entity_fc_layer(e2_h))

        # 拼接向量
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
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