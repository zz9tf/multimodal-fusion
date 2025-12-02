# -*- coding: utf-8 -*-
"""
VAE模型定义
包含Encoder、Decoder和VAE类
"""
import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE编码器，将输入特征编码为潜在空间的均值和方差
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dims (list): 隐藏层维度列表
        latent_dim (int): 潜在空间维度
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = 128):
        """
        初始化编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，默认为[512, 256]
            latent_dim: 潜在空间维度，默认为128
        """
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # 构建编码器网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出层：分别输出均值和log方差
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征张量，形状为 (batch_size, input_dim)
            
        Returns:
            mean: 潜在空间均值，形状为 (batch_size, latent_dim)
            log_var: 潜在空间log方差，形状为 (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE解码器，将潜在向量解码为重构特征
    
    Args:
        latent_dim (int): 潜在空间维度
        hidden_dims (list): 隐藏层维度列表
        output_dim (int): 输出特征维度
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list = None, output_dim: int = None):
        """
        初始化解码器
        
        Args:
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度列表，默认为[256, 512]
            output_dim: 输出特征维度，默认为与latent_dim相同
        """
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        if output_dim is None:
            output_dim = latent_dim
        
        # 构建解码器网络
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在向量，形状为 (batch_size, latent_dim)
            
        Returns:
            x_hat: 重构特征，形状为 (batch_size, output_dim)
        """
        x_hat = self.decoder(z)
        return x_hat


class VAE(nn.Module):
    """
    变分自编码器（VAE）模型
    
    包含编码器和解码器，使用重参数化技巧进行训练
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: str = 'cuda'):
        """
        初始化VAE模型
        
        Args:
            encoder: 编码器模块
            decoder: 解码器模块
            device: 设备（'cuda'或'cpu'）
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()
    
    def _relocate(self):
        """将模型移动到指定设备"""
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def reparameterization(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：从潜在空间采样
        
        Args:
            mean: 潜在空间均值
            log_var: 潜在空间log方差
            
        Returns:
            z: 采样得到的潜在向量
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std).to(self.device)
        z = mean + std * epsilon
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为 (batch_size, input_dim)
            
        Returns:
            x_hat: 重构特征
            z: 潜在向量
            mean: 潜在空间均值
            log_var: 潜在空间log方差
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, z, mean, log_var
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入为潜在向量（用于推理）
        
        Args:
            x: 输入特征
            
        Returns:
            z: 潜在向量
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在向量解码为特征（用于推理）
        
        Args:
            z: 潜在向量
            
        Returns:
            x_hat: 重构特征
        """
        return self.decoder(z)

