import torch
import math
import numpy as np
from transformers import BertModel


model = BertModel.from_pretrained(r"C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week6\demo\bert-base-chinese", return_dict=False)
n = 2                           # 输入最大句子个数    
vocab_size = 21128              # 词表数目
max_sequence_length = 512       # 最大句子长度
embedding_size = 768            # embedding维度
hidden_size = 3072              # 隐藏层维度
num_layers = 12                 # transformer层数

# 手动计算BERT参数量






# BERT模型总参数 = embedding参数 + transformer参数 + pooler参数
# = embedding参数 + self-attention参数 + self-attension_out参数 + feedforward参数 + pool_fc参数


print("模型实际参数量为：", sum(p.numel() for p in model.parameters()))
