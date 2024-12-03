import torch
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载一个预训练的模型
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)  # 将模型迁移到 GPU（如果可用）

# 创建 PEFT 配置
config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA alpha 值
    lora_dropout=0.1,  # LoRA dropout 比例
    target_modules=["transformer.h.*.attn"]  # 适配器作用于所有 Attention 层
)

# 使用 PEFT 加载模型
peft_model = get_peft_model(model, config)

# 简单推理测试
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
outputs = peft_model(**inputs)

print("PEFT model successfully loaded and tested on GPU!")

