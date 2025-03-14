# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config

"""
生成schema文件，事先确定所有的属性类别
不会预测出这之外的属性类别
"""

path = r"../triplet_data/triplet_data.json"
schema = defaultdict(int)
with open(path, "r", encoding="utf8") as f:
    for line in f:
        triplet = json.loads(line)
        attribute = triplet["attribute"]
        schema[attribute] += 1

schema["UNRELATED"] = 1
print("总共有%d个属性"%len(schema))
output = {}
writer = open("schema.json", "w", encoding="utf8")
for index, key in enumerate(schema):
    output[key] = index
writer.write(json.dumps(output, indent=2, ensure_ascii=False))
writer.close()

