# -*- coding: utf-8 -*-
from itertools import combinations

#伪码
#假如一句话中找到多个实体
#两两组合判断关系

entities = NER(sentence)
triplets = []
for enti1, enti2 in combinations(entities, 2):
    relation = RBERT(sentence, enti1, enti2)
    triplets.append([enti1, relation, enti2])