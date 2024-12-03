import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载一个预训练的模型
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)  # 将模型迁移到 GPU（如果可用）

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 示例文本进行推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
outputs = model(**inputs)

print("Transformers model successfully loaded and tested on GPU!")
