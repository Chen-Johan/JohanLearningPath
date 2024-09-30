import cupy as cp
import time

# 检查 GPU 是否可用
if cp.cuda.runtime.getDeviceCount() > 0:
    device_id = cp.cuda.runtime.getDevice()
    device_properties = cp.cuda.runtime.getDeviceProperties(device_id)
    print(f"GPU is available: {device_properties['name']}")
else:
    print("No GPU available")

# 创建大数组并进行操作
x = cp.random.rand(10000, 10000)

# 开始计时
start = time.time()

# 进行矩阵乘法运算
y = cp.dot(x, x)

# 结束计时
end = time.time()

print(f"Time taken for matrix multiplication on GPU: {end - start} seconds")


