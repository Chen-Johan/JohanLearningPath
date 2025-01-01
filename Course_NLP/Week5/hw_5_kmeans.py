#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week5\demo\model.w2v") #加载词向量模型
    sentences = list(load_sentence(r"C:\Users\Gurkha\Desktop\BaDou\Code\Course_NLP\Week5\demo\titles.txt"))  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    
    # for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
    #     sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")
    
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
            
    #计算每个类内到聚类中心的余弦距离
    center_vectors = kmeans.cluster_centers_
    
    cluster_results = []
    for cluster_label, cluster_sentences in sentence_label_dict.items():
        # 提取该聚类的中心向量
        center_vec = center_vectors[cluster_label]
        total_similarity = 0.0
        for single_sentence in cluster_sentences:
            sentence_vec = vectors[sentences.index(single_sentence)]
            # 计算并累加余弦相似度
            total_similarity += np.dot(center_vec, sentence_vec) / (np.linalg.norm(center_vec) * np.linalg.norm(sentence_vec))
        # 求该类的平均相似度
        average_similarity = total_similarity / len(cluster_sentences)
        # 由平均相似度求得平均余弦距离
        average_distance = 1 - average_similarity
        cluster_results.append((cluster_label, average_distance, cluster_sentences))

    # 按平均余弦距离升序排序
    cluster_results.sort(key=lambda x: x[1]) #这里key=lambda x: x[1]的意思是按照第二个元素排序, x[1]是平均余弦距离, x[0]是类别标签, x[2]是类别句子

    # 打印结果
    for label, avg_cos_distance, cluster_sentences in cluster_results:
        print(f"cluster {label} 的类内余弦距离：{avg_cos_distance}")
        for i in range(min(10, len(cluster_sentences))):
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")
    

if __name__ == "__main__":
    main()


# kmeans.cluster_centers_  #聚类中心 42个 128维


# kmeans.labels_ #每个句子的标签 1796个 0-41


# vectors #句子向量 1796个 128维

# 算出每句话到每一个中心的余弦距离，然后再算出每一个类的类内平均距离
# 再按类内平均距离排序，找到类内平均距离比较短的类（观察是不是类内更相似）
