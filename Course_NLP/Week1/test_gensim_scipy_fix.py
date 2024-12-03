'''
为什么要这么做？
这是因为 triu（上三角矩阵）功能是 scipy.linalg 提供的，
但你遇到的错误表明当前的 scipy 版本可能不兼容或者 gensim 没有正确加载 scipy 中的 triu 函数。
因此，我们可以通过从 numpy 导入 triu 来绕过这个问题。

numpy 的 triu 功能是用来生成一个矩阵的上三角部分的，
它与 scipy.linalg.triu 在功能上是一样的。因此，将 triu 从 numpy 导入不会影响功能，且可以绕过当前的兼容性问题。
'''
import gensim
import numpy as np

# 测试 gensim 是否能够正常加载
print("Gensim version:", gensim.__version__)

# 测试 numpy 的 triu 是否正常工作
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.triu(a))