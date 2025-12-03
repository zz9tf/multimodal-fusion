# -*- coding: utf-8 -*-
"""
VAE model definitions.
Contains Encoder, Decoder and VAE classes.
"""
import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE encoder that maps input features into latent mean and log-variance.
    
        Args:
            input_dim (int): input feature dimension
            hidden_dims (list): list of hidden layer dimensions
            latent_dim (int): latent dimension size
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = 128):
        """
        Initialize encoder.

        Args:
            input_dim: input feature dimension
            hidden_dims: hidden layer dimensions, default [512, 256]
            latent_dim: latent dimension, default 128
        """
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build encoder network
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Use GELU activation (often better than ReLU here)
            layers.append(nn.GELU())
            # Use Dropout only on intermediate layers (speed up training)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))  # 减少Dropout比例
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers: mean and log-variance
        self.fc_mean = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: input features of shape (batch_size, input_dim)

        Returns:
            mean: latent mean, shape (batch_size, latent_dim)
            log_var: latent log-variance, shape (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class Decoder(nn.Module):
    """
    VAE decoder that maps latent vectors back to reconstructed features.
    
        Args:
            latent_dim (int): latent space dimension
            hidden_dims (list): list of hidden layer dimensions
            output_dim (int): output feature dimension
    """
    
    def __init__(self, latent_dim: int, hidden_dims: list = None, output_dim: int = None):
        """
        Initialize decoder.

        Args:
            latent_dim: latent space dimension
            hidden_dims: hidden layer dimensions, default [256, 512]
            output_dim: output feature dimension, default = latent_dim
        """
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        if output_dim is None:
            output_dim = latent_dim
        
        # Build decoder network
        layers = []
        prev_dim = latent_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Use GELU activation (often better than ReLU here)
            layers.append(nn.GELU())
            # Use Dropout only on intermediate layers (speed up training)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))  # 减少Dropout比例
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: latent vectors, shape (batch_size, latent_dim)

        Returns:
            x_hat: reconstructed features, shape (batch_size, output_dim)
        """
        x_hat = self.decoder(z)
        return x_hat


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model composed of Encoder and Decoder.
    """
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: str = 'cuda'):
        """
        Initialize VAE model.

        Args:
            encoder: encoder module
            decoder: decoder module
            device: device string, 'cuda' or 'cpu'
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()
    
    def _relocate(self):
        """Move encoder and decoder to the configured device."""
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def reparameterization(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from the latent distribution.

        Args:
            mean: latent mean
            log_var: latent log-variance

        Returns:
            z: sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std).to(self.device)
        z = mean + std * epsilon
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder, reparameterization and decoder.

        Args:
            x: input features, shape (batch_size, input_dim)

        Returns:
            x_hat: reconstructed features
            z: latent vectors
            mean: latent mean
            log_var: latent log-variance
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, z, mean, log_var
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input into latent vectors (for inference).

        Args:
            x: input features

        Returns:
            z: latent vectors
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors back to features (for inference).

        Args:
            z: latent vectors

        Returns:
            x_hat: reconstructed features
        """
        return self.decoder(z)

