# train_vae.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import glob

# ================
# 1. 导入 VAE 模型
# ================
from model import VAE  

# ================
# 2. 参数设置
# ================
DATA_PATH = "/home/amax/ljm/DATA/Desi_galaxyzoo2_fits/DATA/DATAdesi/jpg"
MODEL_DIR = "./models"
RESULT_DIR = "./results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

latent_dim = 128
img_size = 128
batch_size = 32
epochs = 500
lr = 1e-3
max_images = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================
# 3. 数据预处理
# ================
transform = transforms.Compose([
    transforms.CenterCrop(152),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
    # 如果输入是 [0,1]，无需归一化；否则可加 transforms.Normalize(mean=..., std=...)
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
if len(dataset) > max_images:
    dataset = Subset(dataset, list(range(max_images)))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ================
# 4. 模型定义
# ================
model = VAE(latent_dim=latent_dim, img_channels=3, img_size=img_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ================
# 5. 损失函数（重构 + KL）
# ================
def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)  # 可改为 'mean' 或使用 BCE
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# ================
# 6. 训练循环
# ================
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0

    for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        x = x.to(device)
        optimizer.zero_grad()

        # 前向传播
        recon_x, z, mu, log_var = model(x)  # 注意：现在返回四个值
        loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)

        # 反向传播
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()

    avg_loss = epoch_loss / len(dataloader.dataset)
    avg_recon = epoch_recon / len(dataloader.dataset)
    avg_kl = epoch_kl / len(dataloader.dataset)

    print(f"Epoch {epoch+1}, Total Loss: {avg_loss:.6f}, Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}")

    # ================
    # 7. 每个 epoch 保存重建图像
    # ================
    if (epoch + 1) % 1 == 0:
        model.eval()
        with torch.no_grad():
            x_sample = next(iter(dataloader))[0][:4].to(device)  # 取一个 batch 的前4张
            recon_sample, _, _, _ = model(x_sample)

            # 拼接图像：原始 vs 重建
            row_orig = torchvision.utils.make_grid(x_sample, nrow=4, padding=2, normalize=True)
            row_recon = torchvision.utils.make_grid(recon_sample, nrow=4, padding=2, normalize=True)
            comparison = torch.cat([row_orig, row_recon], dim=1)

            plt.figure(figsize=(8, 4))
            plt.imshow(comparison.cpu().permute(1, 2, 0).numpy())
            plt.title(f"Epoch {epoch+1}: Original (top) vs Reconstructed (bottom)", fontsize=10)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/recon_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
        model.train()

# ================
# 8. 保存最终模型
# ================
final_path = f"{MODEL_DIR}/vae_final.pth"
torch.save(model.state_dict(), final_path)
print(f"Final VAE model saved to {final_path}")




# reconstruct.py
import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob
from torchvision import transforms

# ================
# 参数
# ================
MODEL_PATH = "./models/vae_final.pth"
IMAGE_PATHS = glob.glob("/home/amax/ljm/DATA/Desi_galaxyzoo2_fits/DATA/DATAdesi/jpg/1/*.jpg")
if not IMAGE_PATHS:
    raise FileNotFoundError("No JPG images found in the data path.")
IMAGE_PATH_1 = IMAGE_PATHS[0]
RESULT_PATH = "./results/single_recon.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================
# 加载模型
# ================
model = VAE(latent_dim=128, img_channels=3, img_size=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================
# 加载图像并重建
# ================
transform = transforms.Compose([
    transforms.CenterCrop(152),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH_1).convert('RGB')
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    recon_x, z, mu, log_var = model(x)
    recon_x = recon_x.cpu()

# ================
# 可视化
# ================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x[0].cpu().permute(1, 2, 0).numpy())
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(recon_x[0].cpu().permute(1, 2, 0).numpy())
plt.title("Reconstructed (VAE)")
plt.axis('off')

plt.savefig(RESULT_PATH, bbox_inches='tight', dpi=150)
plt.show()
print(f"Reconstruction saved to {RESULT_PATH}")


# interpolate.py
import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob
from torchvision import transforms

# ================
# 参数
# ================
MODEL_PATH = "./models/vae_final.pth"
IMAGE_PATHS = glob.glob("/home/amax/ljm/DATA/Desi_galaxyzoo2_fits/DATA/DATAdesi/jpg/1/*.jpg")
if len(IMAGE_PATHS) < 2:
    raise ValueError("Need at least 2 images for interpolation.")

transform = transforms.Compose([
    transforms.CenterCrop(152),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================
# 加载模型
# ================
model = VAE(latent_dim=128, img_channels=3, img_size=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def load_and_encode(path):
    image = Image.open(path).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, log_var = model.encode(x)
        z = model.reparameterize(mu, log_var)  # 或直接用 mu 做确定性插值
    return z, x

# ================
# 多组插值
# ================
for i in range(10):
    IMAGE_PATH_1 = IMAGE_PATHS[i]
    IMAGE_PATH_2 = IMAGE_PATHS[i + 1]
    RESULT_PATH = f"./results/interpolated_recon_{i+1}.png"

    z1, img1 = load_and_encode(IMAGE_PATH_1)
    z2, img2 = load_and_encode(IMAGE_PATH_2)

    # 中间隐向量（平均）
    z_avg = (z1 + z2) / 2

    # 重建
    with torch.no_grad():
        recon_avg = model.decode(z_avg).cpu()

    # 可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img1[0].cpu().permute(1, 2, 0).numpy())
    plt.title("Galaxy 1")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2[0].cpu().permute(1, 2, 0).numpy())
    plt.title("Galaxy 2")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(recon_avg[0].cpu().permute(1, 2, 0).numpy())
    plt.title("Interpolated Latent → Reconstructed")
    plt.axis('off')

    plt.savefig(RESULT_PATH, bbox_inches='tight', dpi=150)
    plt.close()  # 避免显示
    print(f"Interpolation result saved to {RESULT_PATH}")