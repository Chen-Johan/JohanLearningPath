import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载预训练模型和 tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

# 创建 LoRA 配置，简化为无需指定具体目标模块
config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["classifier"]  # 直接使用 'classifier' 层作为目标模块
)

# 使用 PEFT 加载模型
peft_model = get_peft_model(model, config)

# 简单推理测试
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)

# 执行推理
outputs = peft_model(**inputs)

# 检查是否在 GPU 上
print(f"PEFT model is on device: {peft_model.device}")
print(f"Inputs are on device: {inputs['input_ids'].device}")
print(f"Outputs are on device: {outputs.logits.device}")
print("PEFT model successfully loaded and tested on GPU!")


