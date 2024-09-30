import tensorrt as trt

print(f"TensorRT version: {trt.__version__}")


# 创建 Logger
logger = trt.Logger(trt.Logger.WARNING)

# 创建 Builder
builder = trt.Builder(logger)

# 创建网络定义
network = builder.create_network()

# 打印成功信息
print("TensorRT 基本功能测试通过！")


#另外tensorrt压缩包内包含samples文件夹，里面有很多例子，可以参考