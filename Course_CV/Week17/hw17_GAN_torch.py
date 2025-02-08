import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义超参数
z_dim = 100  # 噪声维度
batch_size = 128
lr = 0.0002  # 学习率
epochs = 50

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个简单的全连接生成器
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)  # 28x28 图片大小，MNIST 数据集

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))  # 输出图像像素值范围在[-1, 1]
        return x

# 创建一个简单的全连接判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input image to a 1D vector
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # 输出为0-1之间的概率值
        return x

# 初始化生成器和判别器
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# 使用 Adam 优化器
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 二进制交叉熵损失函数
criterion = nn.BCELoss()

# 加载 MNIST 数据集（假设你已经下载好了数据集）
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 训练循环
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        # 1. 训练判别器
        real_images = real_images.view(real_images.size(0), -1).to(device)  # Flatten the images

        # 获取当前批次的大小
        batch_size = real_images.size(0)

        # 真实图片标签为1
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 生成随机噪声输入
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)

        # 判别器输出真实图像的判别值
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        # 判别器输出生成图像的判别值
        fake_outputs = discriminator(fake_images.detach())  # .detach() 让生成器不更新
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake

        # 反向传播并更新判别器
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # 2. 训练生成器
        # 生成器试图“欺骗”判别器，因此标签应该是1
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)

        # 反向传播并更新生成器
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')           #Dloss是判别器的损失，Gloss是生成器的损失

    # 每个epoch结束后，生成一些图像进行展示
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)
            fake_images = fake_images.view(fake_images.size(0), 28, 28)
            # 将生成的图像保存为图像文件
            fake_images = fake_images.cpu().numpy()
            num_images = fake_images.shape[0]
            grid_size = int(np.ceil(np.sqrt(num_images)))
            fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            for i in range(grid_size):
                for j in range(grid_size):
                    if i * grid_size + j < num_images:
                        ax[i, j].imshow(fake_images[i * grid_size + j], cmap='gray')
                    ax[i, j].axis('off')
            plt.show()

