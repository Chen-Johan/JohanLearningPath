'''
编辑距离(Levenshtein距离)：是指两个字符串之间，由一个转换成另一个所需的最少编辑操作次数。
'''

import numpy as np

# 计算编辑距离
def calculate_edit_distance(str1, str2):
    # 创建一个二维矩阵，用于存储子问题的解
    dp_matrix = np.zeros((len(str1) + 1, len(str2) + 1))

    # 初始化第一列和第一行
    for i in range(len(str1) + 1):
        dp_matrix[i][0] = i
    for j in range(len(str2) + 1):
        dp_matrix[0][j] = j

    # 填充矩阵，计算编辑距离
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp_matrix[i][j] = min(
                dp_matrix[i - 1][j] + 1,  # 删除操作
                dp_matrix[i][j - 1] + 1,  # 插入操作
                dp_matrix[i - 1][j - 1] + cost  # 替换操作
            )

    # 返回编辑距离（矩阵右下角的值）
    return dp_matrix[len(str1)][len(str2)]

# 基于编辑距离计算字符串的相似度
def similarity_based_on_edit_distance(str1, str2):
    edit_distance = calculate_edit_distance(str1, str2)
    max_length = max(len(str1), len(str2))
    # 计算相似度，返回值在0和1之间，越接近1表示越相似
    return 1 - edit_distance / max_length


if __name__ == "__main__":
    str1 = "kitten"
    str2 = "sitting"
    edit_distance = calculate_edit_distance(str1, str2)
    similarity = similarity_based_on_edit_distance(str1, str2)
    print("编辑距离：", edit_distance)
    print("相似度：", similarity)
    