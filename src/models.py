"""Model architectures for teacher and student networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMTeacher(nn.Module):
    """Teacher model: CNN-LSTM for water quality forecasting."""
    
    def __init__(self, input_features=10, output_features=10, 
                 cnn_channels=[32, 64, 128], lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        
        # CNN layers for local temporal patterns
        self.conv1 = nn.Conv1d(input_features, cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        
        # LSTM for longer-range dependencies
        self.lstm = nn.LSTM(cnn_channels[2], lstm_hidden, lstm_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output head
        self.fc = nn.Linear(lstm_hidden, output_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last time step
        
        x = self.dropout(x)
        out = self.fc(x)
        
        return out


class TCNBlock(nn.Module):
    """Temporal Convolutional Network residual block."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        
        return F.relu(out + residual)


class CNNTCNStudent(nn.Module):
    """Compact student model: CNN-TCN for edge deployment.
    
    Architecture from Table in paper (Section 3.3):
    - Conv layers with increasing dilation
    - TCN blocks with residual connections
    - Global pooling and dense layers
    """
    
    def __init__(self, input_features=10, output_features=10, dropout=0.2):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, dilation=4, padding=4)
        
        # TCN blocks
        self.tcn1 = TCNBlock(64, 64, kernel_size=3, dilation=8, dropout=dropout)
        self.tcn2 = TCNBlock(64, 64, kernel_size=3, dilation=16, dropout=dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # TCN blocks
        x = self.tcn1(x)
        x = self.tcn2(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)  # [batch, 64]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_teacher(config):
    """Factory function to create teacher model from config."""
    arch = config['model']['teacher']['architecture']
    
    if arch == 'CNN-LSTM':
        return CNNLSTMTeacher(
            input_features=len(config['data']['features']),
            output_features=len(config['data']['features']),
            cnn_channels=config['model']['teacher']['cnn_channels'],
            lstm_hidden=config['model']['teacher']['lstm_hidden'],
            lstm_layers=config['model']['teacher']['lstm_layers'],
            dropout=config['model']['teacher']['dropout']
        )
    else:
        raise ValueError(f"Unknown teacher architecture: {arch}")


def create_student(config):
    """Factory function to create student model from config."""
    return CNNTCNStudent(
        input_features=len(config['data']['features']),
        output_features=len(config['data']['features']),
        dropout=config['model']['student'].get('dropout', 0.2)
    )
