import torch
import math
import numpy as np
from transformers import BertModel


model = BertModel.from_pretrained(r"C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week6\demo\bert-base-chinese", return_dict=False)
n = 2                           # 输入最大句子个数    
vocab_size = 21128              # 词表数目
max_sequence_length = 512       # 最大句子长度
embedding_size = 768            # embedding维度
intermediate_size = 3072        # FeedForward层维度
hidden_size = 768               # 隐藏层维度
num_layers = 12                 # transformer层数

# 手动计算BERT参数量
"""
odict_keys
(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight', 
'embeddings.token_type_embeddings.weight', 'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias', 


'encoder.layer.0.attention.self.query.weight', 'encoder.layer.0.attention.self.query.bias', 
'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias', 
'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias',
 
'encoder.layer.0.attention.output.dense.weight', 'encoder.layer.0.attention.output.dense.bias', 

'encoder.layer.0.attention.output.LayerNorm.weight', 'encoder.layer.0.attention.output.LayerNorm.bias', 

'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias', 

'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias', 

'encoder.layer.0.output.LayerNorm.weight', 'encoder.layer.0.output.LayerNorm.bias', 


'pooler.dense.weight', 'pooler.dense.bias'])
"""

# Embedding参数量 = 词表数目 * embedding维度 + 最大句子长度 * embedding维度 + 输入最大句子个数 * embedding维度 + 2 * embedding维度 + embedding维度
word_embeddings = vocab_size * embedding_size # 词表数目 * embedding维度 = 21128 * 768
position_embeddings = max_sequence_length * embedding_size # 最大句子长度 * embedding维度 = 512 * 768
token_type_embeddings = n * embedding_size # 输入最大句子个数 * embedding维度 = 2 * 768
LayerNorm_w = 2 * embedding_size # 2 * embedding维度 = 2 * 768 在 BERT 的 Embeddings 中，LayerNorm 只需要 gamma 和 beta 两个向量

embedding_parameters = word_embeddings + position_embeddings + token_type_embeddings + LayerNorm_w


# Self-Attention参数量，其中embedding_size * embedding_size为QKV的参数量，embedding_size为QKV的bias参数量
self_attention_parameters = (embedding_size * embedding_size + embedding_size)*3 # QKV = embedding_size * embedding_size + embedding_size

# Self-Attention_out参数量，其中embedding_size * embedding_size为W的参数量，embedding_size为bias的参数量
self_attention_out_parameters = embedding_size * embedding_size + embedding_size # W = embedding_size * embedding_size + embedding_size

# =归一化参数量，其中embedding_size为gamma和beta的参数量 γ（gamma）：用于缩放归一化后的输出。β（beta）：用于平移归一化后的输出。
LayerNorm_parameters = 2 * embedding_size # 2 * embedding_size = 2 * 768


# FeedForward参数量，
# 第一层（Intermediate 层）将输入从 hidden size 映射到 intermediate size。
#   其中[hidden_size × intermediate_size + intermediate_size]
# 第二层（Output 层）将 intermediate size 映射回 hidden size。
#   其中[intermediate_size × hidden_size + hidden_size]
feedforward_parameters = (hidden_size * intermediate_size + intermediate_size) + (intermediate_size * hidden_size + hidden_size) # (768 * 3072 + 3072) + (3072 * 768 + 768)

# 归一化参数量，其中embedding_size为gamma和beta的参数量
LayerNorm_parameters = 2 * embedding_size # 2 * embedding_size = 2 * 768


# Pooler参数量 = hidden_size * hidden_size 将整个序列的信息汇总成一个固定大小的向量
pooler_parameters = embedding_size * embedding_size + embedding_size


# BERT模型总参数 = Embedding参数量 + Self-Attention参数量 + Self-Attention_out参数量 + 归一化参数量 + FeedForward参数量 + 归一化参数量 + Pooler参数
total_parameters = embedding_parameters + num_layers * (self_attention_parameters + self_attention_out_parameters + LayerNorm_parameters + feedforward_parameters + LayerNorm_parameters) + pooler_parameters

print("手动计算BERT参数量为: ", total_parameters)
print("模型实际参数量为：", sum(p.numel() for p in model.parameters()))
