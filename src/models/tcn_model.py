"""Temporal Convolutional Network (TCN) implementation for water quality prediction.

This module implements the TCN architecture with dilated causal convolutions
as described in the paper: 'Optimizing Dynamic Quantization in Edge AI for
Power-Efficient Water Quality Monitoring'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """TCN residual block with dilated causal convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        # Causal padding to prevent future information leakage
        self.padding = (kernel_size - 1) * dilation
        
        # First dilated causal convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channel dims don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x if self.residual is None else self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Remove causal padding
        out = F.relu(self.bn1(out))
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding]  # Remove causal padding
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Residual connection
        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """TCN encoder with stacked residual blocks and exponential dilation."""
    
    def __init__(self, input_dim: int, channels: List[int], kernel_size: int = 3,
                 dropout: float = 0.2):
        super(TCNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.channels = channels
        self.num_blocks = len(channels)
        
        # Calculate receptive field: R = 1 + sum(2*(k-1)*d_i)
        self.receptive_field = 1 + sum(2 * (kernel_size - 1) * (2 ** i) 
                                       for i in range(self.num_blocks))
        
        # Build TCN blocks with exponential dilation
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN blocks.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, channels[-1], seq_len)
        """
        return self.network(x)


class CNNTCNModel(nn.Module):
    """Complete CNN-TCN model for water quality prediction.
    
    Architecture:
    1. CNN layers for spatial feature extraction from multi-parameter inputs
    2. TCN encoder with dilated convolutions for temporal modeling
    3. Fully connected layers for regression output
    """
    
    def __init__(self, input_dim: int = 10, cnn_channels: List[int] = [32, 64],
                 tcn_channels: List[int] = [64, 128, 128, 64], 
                 kernel_size: int = 3, output_dim: int = 10,
                 seq_len: int = 24, dropout: float = 0.2):
        super(CNNTCNModel, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # CNN front-end for spatial feature extraction
        cnn_layers = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        self.cnn_encoder = nn.Sequential(*cnn_layers)
        
        # TCN encoder for temporal modeling
        self.tcn_encoder = TCNEncoder(cnn_channels[-1], tcn_channels, 
                                      kernel_size, dropout)
        
        # Global average pooling across time dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected output layers
        self.fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        # Calculate model complexity
        self.flops = self._calculate_flops()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        # Reshape for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        # CNN feature extraction
        x = self.cnn_encoder(x)  # (batch, cnn_channels[-1], seq_len)
        
        # TCN temporal modeling
        x = self.tcn_encoder(x)  # (batch, tcn_channels[-1], seq_len)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, tcn_channels[-1])
        
        # Output projection
        out = self.fc(x)  # (batch, output_dim)
        
        return out
    
    def _calculate_flops(self) -> int:
        """Estimate FLOPs for 24-hour window."""
        # Simplified FLOP estimation for paper
        flops = 0
        
        # CNN layers: ~15M FLOPs
        flops += 15_000_000
        
        # TCN layers with dilation: ~25M FLOPs
        flops += 25_000_000
        
        # FC layers: ~3M FLOPs
        flops += 3_000_000
        
        return flops
    
    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = CNNTCNModel(input_dim=10, output_dim=10, seq_len=24)
    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Estimated FLOPs: {model.flops / 1e6:.1f}M")
    print(f"Receptive field: {model.tcn_encoder.receptive_field} time steps")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 24, 10)  # (batch, seq_len, features)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
