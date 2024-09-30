import tensorflow as tf
import timeit

# 创建一个大的矩阵计算任务
size = 10000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# 在CPU上运行矩阵乘法
def cpu_run():
    with tf.device("/CPU:0"):
        c = tf.matmul(a, b)
    return c

# 在GPU上运行矩阵乘法
def gpu_run():
    with tf.device("/GPU:0"):
        c = tf.matmul(a, b)
    return c

# 计算CPU和GPU的执行时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")