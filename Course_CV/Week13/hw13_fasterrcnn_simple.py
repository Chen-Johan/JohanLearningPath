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

model.eval()

# 优化：定义device，明确注释
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


# 加载图像并进行预处理
def preprocess_image(input_image):
    """
    将输入图像转换为Tensor并添加Batch维度
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    return transform(input_image).unsqueeze(0)  # 添加batch维度

# 进行推理
def detect_objects(input_image_path):
    """
    执行目标检测推理并返回结果
    """
    image = Image.open(input_image_path).convert("RGB")
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        detections = model(input_tensor)

    return detections

# 显示结果
def draw_detection_results(input_image, detections):
    """
    在图像上绘制检测结果并显示
    """
    pred_boxes = detections[0]['boxes'].cpu().numpy()
    pred_labels = detections[0]['labels'].cpu().numpy()
    pred_scores = detections[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(input_image)

    detection_threshold = 0.5
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score > detection_threshold:
            box_top_left = (box[0], box[1])
            box_bottom_right = (box[2], box[3])
            draw.rectangle([box_top_left, box_bottom_right], outline='red', width=2)
            draw.text((box[0], box[1] - 10), f"{label}", fill='red')

    input_image.show()


def main():
    """
    入口函数：选择测试图像并执行推理
    """
    test_image_path = r'C:\Users\Gurkha\Desktop\BaDou\Code\Course_CV\Week13\demo\fasterrcnn简单版\street.jpg'  # 替换为你的图像路径
    detections = detect_objects(test_image_path)
    image = Image.open(test_image_path)
    draw_detection_results(image, detections)



# 使用示例
if __name__ == '__main__':
    main()
