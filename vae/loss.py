# -*- coding: utf-8 -*-
"""
VAE损失函数
包含KLD（Kullback-Leibler Divergence）损失计算

根据论文公式：
- L_MSE = (1/N) * sum ||f_p^(i) - Dec_VAE(Enc_VAE(f_p^(i)))||^2
- L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
- L_VAE = L_MSE + L_KLD
"""
import torch
from typing import Tuple


def compute_kld_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    计算Kullback-Leibler散度损失（KLD Loss）
    
    公式：L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
    其中 sigma_j^2 = exp(log_var_j), mu_j 是均值
    
    这个损失函数鼓励潜在空间分布接近标准正态分布 N(0, I)
    
    Args:
        mean: 潜在空间均值，形状为 (batch_size, latent_dim)
        log_var: 潜在空间log方差，形状为 (batch_size, latent_dim)
        
    Returns:
        KLD损失值（标量张量）
    """
    # 计算KLD: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    # 等价于: -0.5 * sum(1 + log_var - mean^2 - var)
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    # 对所有样本和维度求平均
    kld = kld.mean()
    return kld

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, 
             log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算VAE总损失
    
    根据论文，使用KLD损失（标准VAE损失函数）：
    - L_MSE: 重构损失（MSE）
    - L_KLD: KL散度损失，使潜在空间分布接近标准正态分布
    - L_VAE = L_MSE + L_KLD
    
    Args:
        x: 原始输入特征，形状为 (batch_size, feature_dim)
        x_hat: 重构特征，形状为 (batch_size, feature_dim)
        mean: 潜在空间均值，形状为 (batch_size, latent_dim)
        log_var: 潜在空间log方差，形状为 (batch_size, latent_dim)
        
    Returns:
        total_loss: 总损失
        recon_loss: 重构损失（MSE）
        reg_loss: 正则化损失（KLD）
    """
    # 重构损失（MSE）
    # L_MSE = (1/N) * sum ||f_p^(i) - Dec_VAE(Enc_VAE(f_p^(i)))||^2
    recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    
    # 使用KLD损失
    # L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
    reg_loss = compute_kld_loss(mean, log_var)
    
    # 总损失
    total_loss = recon_loss + reg_loss
    
    return total_loss, recon_loss, reg_loss

