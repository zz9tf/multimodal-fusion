# -*- coding: utf-8 -*-
"""
VAE loss functions.
Includes KLD (Kullback-Leibler Divergence) loss computation.

Formulas from the paper:
- L_MSE = (1/N) * sum ||f_p^(i) - Dec_VAE(Enc_VAE(f_p^(i)))||^2
- L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
- L_VAE = L_MSE + L_KLD
"""
import torch
from typing import Tuple


def compute_kld_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence loss (KLD loss).

    Formula: L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
    where sigma_j^2 = exp(log_var_j), and mu_j is the mean.

    This loss encourages the latent distribution to approach N(0, I).

    Args:
        mean: latent mean, shape (batch_size, latent_dim)
        log_var: latent log-variance, shape (batch_size, latent_dim)
        
    Returns:
        KLD loss value (scalar tensor)
    """
    # KLD: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    # Equivalent to: -0.5 * sum(1 + log_var - mean^2 - var)
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    # 对所有样本和维度求平均
    kld = kld.mean()
    return kld

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, 
             log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the total VAE loss.

    According to the paper, we use the standard VAE loss:
    - L_MSE: reconstruction loss (MSE)
    - L_KLD: KL divergence loss, pushing the latent towards N(0, I)
    - L_VAE = L_MSE + L_KLD

    Args:
        x: original input features, shape (batch_size, feature_dim)
        x_hat: reconstructed features, shape (batch_size, feature_dim)
        mean: latent mean, shape (batch_size, latent_dim)
        log_var: latent log-variance, shape (batch_size, latent_dim)
        
    Returns:
        total_loss: total loss
        recon_loss: reconstruction loss (MSE)
        reg_loss: regularization loss (KLD)
    """
    # Reconstruction loss (MSE)
    # L_MSE = (1/N) * sum ||f_p^(i) - Dec_VAE(Enc_VAE(f_p^(i)))||^2
    recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    
    # KLD loss
    # L_KLD = -(1/2) * sum_j (1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
    reg_loss = compute_kld_loss(mean, log_var)
    
    # Total loss
    total_loss = recon_loss + reg_loss
    
    return total_loss, recon_loss, reg_loss

