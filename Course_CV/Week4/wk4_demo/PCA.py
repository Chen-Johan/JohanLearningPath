#!/usr/bin/env python
# encoding=utf-8

# Iris data set example
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的特征数据，y表示数据标签
pca = dp.PCA(n_components=2)  # 使用pca算法，并且将维度降低到2
reduced_x = pca.fit_transform(x)  # 将原始数据进行降维，保存在reduced_x中

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):  # 遍历所有数据点，将降维后的数据点保存在不同的类别中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()