# Autoencoder_model.py - 实际上是 Autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F

# vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, img_size=64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          # 8 -> 4
            nn.ReLU(),
        )
        
        # 潜在空间映射
        self.fc_encode = nn.Linear(256 * 8 * 8, latent_dim)      


        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1), # 32 -> 64
            # nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        return z  # 只返回 z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x, z  # 不返回 mu/logvar
    
def Autoencoder_loss(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    return recon_loss


# vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, img_size=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),            # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),           # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),          # 8 -> 4
            nn.ReLU(),
        )

        # 输出 mu 和 log_var
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)      # 注意：这里是 4x4，不是 8x8！
        self.fc_log_var = nn.Linear(256 * 8 * 8, latent_dim)

        # 解码器输入映射
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1), # 32 -> 64
            # 可选：加 Sigmoid 如果输入归一化到 [0,1]
            # nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 标准差
        eps = torch.randn_like(std)     # 从标准正态采样
        return mu + eps * std           # 重参数化

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)  # 注意恢复为 4x4 特征图
        return self.decoder(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, z, mu, log_var

import torch.nn.functional as F

def vae_loss(recon_x, x, mu, log_var, reduction='sum'):
    """
    VAE 损失函数
    
    Args:
        recon_x: 重建图像 [B, C, H, W]
        x: 原始输入 [B, C, H, W]
        mu: 均值 [B, latent_dim]
        log_var: 对数方差 [B, latent_dim]
        reduction: sum, mean, none
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # 重构损失：MSE 
    # 选择其一：
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    
    if reduction == 'sum':
        kl_loss = kl_loss.sum()
    elif reduction == 'mean':
        kl_loss = kl_loss.mean()

    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss