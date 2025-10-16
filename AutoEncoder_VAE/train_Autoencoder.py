
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision

from Autoencoder_model import Autoencoder  # 注意类名

# 参数
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

# 数据
transform = transforms.Compose([
    transforms.CenterCrop(152),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
if len(dataset) > max_images:
    dataset = Subset(dataset, list(range(max_images)))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型
model = Autoencoder(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # 仅使用 MSE

# 训练
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for x, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        x = x.to(device)
        optimizer.zero_grad()
        recon_x, z = model(x)  # 不再有 mu/logvar
        loss = criterion(recon_x, x)  # 只计算重建损失
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    # 每1个epoch保存可视化结果
    if (epoch + 1) % 1 == 0:
        model.eval()
        with torch.no_grad():
            x_sample = x[:4].to(device)
            recon_sample, _ = model(x_sample)

            # 使用 make_grid 拼接
            row_orig = torchvision.utils.make_grid(x_sample, nrow=4, padding=2, normalize=True)
            row_recon = torchvision.utils.make_grid(recon_sample, nrow=4, padding=2, normalize=True)
            comparison = torch.cat([row_orig, row_recon], dim=1)  # 上下拼接

            plt.figure(figsize=(8, 4))
            plt.imshow(comparison.cpu().permute(1, 2, 0).numpy())
            plt.title(f"Epoch {epoch+1}: Original (top) vs Reconstructed (bottom)", fontsize=10)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/recon_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
        model.train()


    # 最终保存
    final_path = f"{MODEL_DIR}/autoencoder_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import glob  # 开头导入
    from torchvision import transforms


    # 参数
    MODEL_PATH = "./models/autoencoder_final.pth"
    IMAGE_PATHS = glob.glob("/home/amax/ljm/DATA/Desi_galaxyzoo2_fits/DATA/DATAdesi/jpg/1/*.jpg")
    if not IMAGE_PATHS:
        raise FileNotFoundError("No JPG images found in the data path.")
    IMAGE_PATH_1 = IMAGE_PATHS[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


    # 加载图像
    image = Image.open(IMAGE_PATH_1).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        recon_x, z = model(x)
        recon_x = recon_x.cpu()

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].cpu().permute(1, 2, 0).numpy())
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_x[0].cpu().permute(1, 2, 0).numpy())
    plt.title("Reconstructed")
    plt.axis('off')

    RESULT_PATH = "./results/single_recon.png"

    plt.savefig(RESULT_PATH, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Reconstruction saved to {RESULT_PATH}")



    # test_interpolate.py
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import glob
    from torchvision import transforms


    # 参数
    MODEL_PATH = "./models/autoencoder_final.pth"
    IMAGE_PATHS = glob.glob("/home/amax/ljm/DATA/Desi_galaxyzoo2_fits/DATA/DATAdesi/jpg/1/*.jpg")
    if len(IMAGE_PATHS) < 2:
        raise ValueError("Need at least 2 images for interpolation.")

    for i in range(1,10):
        IMAGE_PATH_1 = IMAGE_PATHS[i+1]
        IMAGE_PATH_2 = IMAGE_PATHS[i]

        RESULT_PATH = "./results/interpolated_recon_" + str(i) + ".png"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()


        def load_and_encode(path):
            image = Image.open(path).convert('RGB')
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model.encode(x)
            return z, x

        # 编码两张图
        z1, img1 = load_and_encode(IMAGE_PATH_1)
        z2, img2 = load_and_encode(IMAGE_PATH_2)

        # 特征均值
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
        plt.title("Averaged Latent → Reconstructed")
        plt.axis('off')

        plt.savefig(RESULT_PATH, bbox_inches='tight', dpi=150)
        plt.show()
        print(f"Interpolation result saved to {RESULT_PATH}")