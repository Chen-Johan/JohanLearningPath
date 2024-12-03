import torch

# 检查 Flash Attention 是否已启用
print(f"Flash Attention enabled: {torch.backends.cuda.is_built_with_flash_attention()}")


