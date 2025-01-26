import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline


def main():
    # 尝试加载 Stable Diffusion 模型
    try:
        # 从预训练的存储库加载 Stable Diffusion 模型，使用 float16 数据类型以节省内存
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        # 将模型移动到 GPU 上，如果 GPU 可用，否则使用 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe.to(device)
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    # 定义生成图像的提示词
    prompt = "A beautiful sunset over the ocean"
    try:
        # 使用自动混合精度进行推理
        with autocast(device):
            # 生成图像
            image = pipe(prompt).images[0]
        # 保存生成的图像
        image.save("generated_image.png")
        print("Image generated and saved successfully.")
    except Exception as e:
        print(f"Failed to generate the image: {e}")


if __name__ == "__main__":
    main()
