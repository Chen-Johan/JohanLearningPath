import torch

# 檢查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用GPU
else:
    device = torch.device("cpu")  # 使用CPU