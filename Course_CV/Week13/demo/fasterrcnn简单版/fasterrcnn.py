import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image,ImageDraw
import numpy as np

# 加载预训练模型
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# 从 torchvision 0.13 起，pretrained 参数已被弃用，需要使用 weights 参数来指定预训练模型，
# 让模型在最新版本中保持兼容性。这样修改是遵循了新API规范。
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
#
model.eval()

# 如果你的模型是在GPU上训练的，确保模型也在GPU上进行推理
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 加载图像并进行预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 添加batch维度

# 进行推理
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction

# 显示结果
def show_result(image, prediction):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)
    #image = Image.fromarray(np.uint8(image))

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # 阈值可根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2],box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            print(str(label))
            draw.text((box[0], box[1] - 10), str(label), fill='red')
    image.show()

# 使用示例
image_path = r'C:\Users\Gurkha\Desktop\BaDou\Code\Course_CV\Week13\demo\fasterrcnn简单版\street.jpg'  # 替换为你的图像路径
prediction = infer(image_path)
image = Image.open(image_path)
image = show_result(image, prediction)
