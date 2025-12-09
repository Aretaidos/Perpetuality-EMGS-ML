# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FIXED Neural Network Architectures for Isolated Channel sEMG Gesture Recognition.

KEY FIXES:
1. M1 (CNN+LSTM): Proper input_channels=7 default, correct left_context/stride
2. M2 (CNN-only): Fixed temporal alignment, improved architecture
3. Both: GPU-optimized with proper weight initialization

Expected Performance (7 channels, 9 gestures):
- M1 (CNN+LSTM): 90-95% accuracy
- M2 (CNN-only): 88-93% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReinhardCompression(nn.Module):
    """Dynamic range compression using Reinhard operator.

    Normalizes EMG amplitude: output = range * x / (midpoint + |x|)
    """

    def __init__(self, range_val: float = 1.0, midpoint: float = 32.0):
        super().__init__()
        self.range_val = range_val
        self.midpoint = midpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.range_val * x / (self.midpoint + torch.abs(x))


class FixedDiscreteGesturesLSTM(nn.Module):
    """
    FIXED CNN+LSTM Architecture for 7-channel sEMG gesture recognition (M1).

    Architecture:
    1. Amplitude normalization (Reinhard compression)
    2. Conv1D: 7 → conv_output_channels, downsamples 2kHz→200Hz
    3. LayerNorm + Dropout
    4. Stacked LSTM (3 layers)
    5. LayerNorm
    6. Linear projection → 9 gesture classes

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 7
    conv_output_channels : int
        Number of convolutional output channels. Default: 128
    kernel_width : int
        Convolutional kernel width. Default: 15
    stride : int
        Convolutional stride. Default: 10
    lstm_hidden_size : int
        LSTM hidden state size. Default: 128
    lstm_num_layers : int
        Number of stacked LSTM layers. Default: 3
    output_channels : int
        Number of output gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.1
    """

    def __init__(
        self,
        input_channels: int = 7,  # FIXED: Default to 7
        conv_output_channels: int = 128,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        # Temporal alignment parameters
        self.left_context = kernel_width - 1
        self.stride = stride

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial Conv1D
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )

        # 3. Post-conv processing
        self.conv_ln = nn.LayerNorm(conv_output_channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.1)

        # 4. Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        # 5. Post-LSTM normalization
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 6. Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Conv layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # LSTM: Orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 7, time)

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, time')
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Conv1D
        x = self.conv(x)  # (B, 128, T')
        x = self.relu(x)
        x = self.conv_dropout(x)

        # 3. Transpose for LayerNorm
        x = x.transpose(1, 2)  # (B, T', 128)
        x = self.conv_ln(x)

        # 4. LSTM
        x = x.contiguous()
        x, _ = self.lstm(x)  # (B, T', hidden)

        # 5. Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # 6. Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # 7. Transpose back
        x = x.transpose(1, 2)  # (B, 9, T')

        return x


class InceptionBlock1D(nn.Module):
    """Multi-scale 1D Inception block for temporal pattern capture."""

    def __init__(self, in_channels: int, out_channels_per_path: int = 64):
        super().__init__()

        # Short-term path (2.5ms at 2kHz)
        self.conv_short = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_path, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels_per_path),
            nn.ReLU(inplace=True),
        )

        # Medium-term path (7.5ms)
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_path, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels_per_path),
            nn.ReLU(inplace=True),
        )

        # Long-term path (12.5ms)
        self.conv_long = nn.Sequential(
            nn.Conv1d(in_channels, out_channels_per_path, kernel_size=25, padding=12),
            nn.BatchNorm1d(out_channels_per_path),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.conv_short(x)
        medium = self.conv_medium(x)
        long = self.conv_long(x)
        return torch.cat([short, medium, long], dim=1)


class FixedDiscreteGesturesCNN(nn.Module):
    """
    FIXED Pure CNN Architecture for 7-channel sEMG gesture recognition (M2).

    Architecture:
    1. Amplitude normalization
    2. Initial Conv Block: 7 → 64, downsample to 200Hz
    3. Inception Block: Multi-scale features
    4. Conv Block 2: 192 → 128
    5. Conv Block 3: 128 → 128
    6. Output projection → 9 classes

    Parameters
    ----------
    input_channels : int
        Number of input channels. Default: 7
    output_channels : int
        Number of gesture classes. Default: 9
    """

    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 9,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Temporal alignment
        self.left_context = 20
        self.stride = 40

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial conv (2kHz → 200Hz)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=21, stride=10, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 3. Inception block
        self.inception = InceptionBlock1D(64, out_channels_per_path=64)

        # 4. Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 5. Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 6. Refinement conv
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # 7. Output projection
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Conv1d(256, output_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (B, 7, T)

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (B, 9, T')
        """
        x = self.compression(x)
        x = self.conv1(x)      # (B, 64, T/10)
        x = self.inception(x)  # (B, 192, T/10)
        x = self.conv2(x)      # (B, 128, T/20)
        x = self.conv3(x)      # (B, 128, T/40)
        x = self.conv4(x)      # (B, 256, T/40)
        x = self.dropout(x)
        x = self.fc_out(x)     # (B, 9, T/40)

        return x


def create_gesture_model(
    model_type: str = 'lstm',
    input_channels: int = 7,
    output_channels: int = 9,
    **kwargs
) -> nn.Module:
    """
    Factory function to create gesture recognition models.

    Parameters
    ----------
    model_type : str
        'lstm' for M1, 'cnn' for M2, 'm1_compressed' for XIAO-optimized M1
    input_channels : int
        Number of EMG channels (default: 7)
    output_channels : int
        Number of gesture classes (default: 9)
    **kwargs
        Additional model-specific parameters

    Returns
    -------
    nn.Module
        Configured model
    """
    if model_type.lower() in ['lstm', 'm1', 'cnn_lstm']:
        return FixedDiscreteGesturesLSTM(
            input_channels=input_channels,
            output_channels=output_channels,
            **kwargs
        )
    elif model_type.lower() in ['cnn', 'm2', 'cnn_only']:
        return FixedDiscreteGesturesCNN(
            input_channels=input_channels,
            output_channels=output_channels,
        )
    elif model_type.lower() in ['m1_compressed', 'compressed', 'xiao']:
        return CompressedM1ForXIAO(
            input_channels=input_channels,
            output_channels=output_channels,
            **kwargs
        )
    elif model_type.lower() in ['m1_4channel', '4channel', 'tinyml', 'm1_4ch']:
        return M1_4Channel_TinyML(
            input_channels=input_channels,
            output_channels=output_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class CompressedM1ForXIAO(nn.Module):
    """
    Compressed CNN+LSTM Architecture for XIAO nRF52840 TinyML Deployment.

    Memory-optimized version of FixedDiscreteGesturesLSTM (M1) that fits in
    256KB RAM of the nRF52840 microcontroller.

    Original M1:
        Conv(7→128) → LSTM(128, 3 layers) → FC(9)
        Parameters: 411,529 | RAM: ~398KB (DOES NOT FIT)

    Compressed M1:
        Conv(7→64) → LSTM(32, 1 layer) → FC(9)
        Parameters: ~45,000 | RAM: ~80-100KB (FITS)

    Memory Budget on XIAO nRF52840:
        - Tensor Arena: 80KB
        - Input Buffer (7×1000×4): 28KB
        - TFLite Runtime: 25KB
        - LSTM States: 0.3KB
        - Stack/Heap: 40KB
        - Total: ~173KB (fits in 220KB usable)

    Target Hardware:
        - Seeed Studio XIAO nRF52840 (non-Sense)
        - nRF52840: ARM Cortex-M4F @ 64MHz
        - RAM: 256KB total, ~220KB usable
        - Flash: 1MB + 2MB onboard

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 7
    conv_output_channels : int
        Number of convolutional output channels. Default: 64 (reduced from 128)
    kernel_width : int
        Convolutional kernel width. Default: 15
    stride : int
        Convolutional stride. Default: 10 (downsample 2kHz→200Hz)
    lstm_hidden_size : int
        LSTM hidden state size. Default: 32 (reduced from 128)
    lstm_num_layers : int
        Number of stacked LSTM layers. Default: 1 (reduced from 3)
    output_channels : int
        Number of output gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.1
    """

    def __init__(
        self,
        input_channels: int = 7,
        conv_output_channels: int = 64,  # REDUCED from 128
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 32,      # REDUCED from 128
        lstm_num_layers: int = 1,        # REDUCED from 3
        output_channels: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_output_channels = conv_output_channels

        # Temporal alignment parameters
        self.left_context = kernel_width - 1
        self.stride = stride

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial Conv1D - reduced channels
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )

        # 3. Post-conv processing
        self.conv_ln = nn.LayerNorm(conv_output_channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.1)

        # 4. Single-layer LSTM (reduced complexity for TinyML)
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0,  # No dropout for single layer
            bidirectional=False,
        )

        # 5. Post-LSTM normalization
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 6. Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Conv layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # LSTM: Orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 7, time)
            For 500ms window: (batch, 7, 1000) at 2kHz

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, time')
            For 1000 samples: (batch, 9, ~100) after stride=10
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Conv1D
        x = self.conv(x)  # (B, 64, T')
        x = self.relu(x)
        x = self.conv_dropout(x)

        # 3. Transpose for LayerNorm
        x = x.transpose(1, 2)  # (B, T', 64)
        x = self.conv_ln(x)

        # 4. LSTM
        x = x.contiguous()
        x, _ = self.lstm(x)  # (B, T', 32)

        # 5. Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # 6. Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # 7. Transpose back
        x = x.transpose(1, 2)  # (B, 9, T')

        return x

    def get_memory_estimate(self) -> dict:
        """
        Estimate memory usage for TinyML deployment.

        Returns
        -------
        dict
            Memory estimates in bytes
        """
        # Parameter count
        params = sum(p.numel() for p in self.parameters())

        # Estimate for INT8 quantization
        model_size_int8 = params  # 1 byte per param

        # Runtime memory (activations, states)
        # For 1000 sample input at stride 10 → 100 timesteps
        window_samples = 1000
        timesteps = window_samples // self.stride

        activation_mem = (
            7 * window_samples * 4 +                    # Input buffer
            self.conv_output_channels * timesteps * 4 + # Conv output
            self.lstm_hidden_size * timesteps * 4 +     # LSTM output
            self.lstm_hidden_size * 2 * 4 +             # LSTM states (h, c)
            9 * timesteps * 4                           # Output
        )

        return {
            'parameters': params,
            'model_size_fp32': params * 4,
            'model_size_int8': model_size_int8,
            'activation_memory': activation_mem,
            'estimated_arena': activation_mem + 20000,  # +20KB overhead
            'fits_nrf52840': (activation_mem + model_size_int8 + 50000) < 220000
        }


class M1_4Channel_TinyML(nn.Module):
    """
    Optimized 4-Channel CNN+LSTM for XIAO nRF52840 TinyML Deployment.

    This architecture is designed for the Seeed Studio XIAO nRF52840 (non-Sense)
    with only 4 EMG channels due to ADC limitations.

    Key Design Decisions:
    - 4 channels: [Ch7, Ch8, Ch13, Ch15] = indices [6, 7, 12, 14] (0-based)
      - 2 flexors (press detection) + 2 extensors (release detection)
      - 66.2% of total channel importance captured
    - 2-layer LSTM (not 1 layer) to maintain hierarchical temporal learning
    - Larger conv filters and LSTM hidden size than CompressedM1 for better accuracy
    - Still fits in 256KB RAM with INT8 quantization

    Architecture:
        Conv1d(4, 72, k=15, s=10) → BatchNorm → ReLU → Dropout(0.3)
        → LSTM(72, 48, num_layers=2) → LayerNorm → FC(9)

    Memory Estimates (INT8 quantized, 2000 sample window):
        - Parameters: ~47,300
        - Model size: ~47KB
        - Activation memory: ~130KB
        - Total runtime: ~208KB (fits in 220KB usable)

    Target Hardware:
        - Seeed Studio XIAO nRF52840 (non-Sense)
        - ARM Cortex-M4F @ 64MHz
        - RAM: 256KB total, ~220KB usable
        - Flash: 1MB + 2MB onboard

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 4
    conv_output_channels : int
        Number of convolutional output channels. Default: 72
    kernel_width : int
        Convolutional kernel width. Default: 15 (7.5ms at 2kHz)
    stride : int
        Convolutional stride. Default: 10 (downsample 2kHz→200Hz)
    lstm_hidden_size : int
        LSTM hidden state size. Default: 48 (reduced for memory fit)
    lstm_num_layers : int
        Number of stacked LSTM layers. Default: 2 (for hierarchical learning)
    output_channels : int
        Number of output gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.3
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv_output_channels: int = 72,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 48,  # Reduced from 56 for memory fit
        lstm_num_layers: int = 2,
        output_channels: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_output_channels = conv_output_channels

        # Temporal alignment parameters
        self.left_context = kernel_width - 1
        self.stride = stride

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial Conv1D with BatchNorm for better convergence
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )
        self.conv_bn = nn.BatchNorm1d(conv_output_channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 3. Two-layer LSTM for hierarchical temporal patterns
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        # 4. Post-LSTM normalization
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 5. Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Conv layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # LSTM: Orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 4, time)
            For 1.25s window: (batch, 4, 2500) at 2kHz

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, time')
            For 2500 samples: (batch, 9, ~250) after stride=10
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Conv1D with BatchNorm
        x = self.conv(x)  # (B, 72, T')
        x = self.conv_bn(x)
        x = self.relu(x)
        x = self.conv_dropout(x)

        # 3. Transpose for LSTM (batch_first=True)
        x = x.transpose(1, 2)  # (B, T', 72)

        # 4. LSTM (2 layers)
        x = x.contiguous()
        x, _ = self.lstm(x)  # (B, T', 56)

        # 5. Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # 6. Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # 7. Transpose back to (B, C, T) format
        x = x.transpose(1, 2)  # (B, 9, T')

        return x

    def get_memory_estimate(self, window_samples: int = 2500) -> dict:
        """
        Estimate memory usage for TinyML deployment.

        Parameters
        ----------
        window_samples : int
            Number of input samples per window. Default: 2500 (1.25s at 2kHz)

        Returns
        -------
        dict
            Memory estimates in bytes
        """
        # Parameter count
        params = sum(p.numel() for p in self.parameters())

        # Estimate for INT8 quantization
        model_size_int8 = params  # 1 byte per param

        # Runtime memory (activations, states)
        timesteps = window_samples // self.stride

        activation_mem = (
            self.input_channels * window_samples * 4 +      # Input buffer (fp32)
            self.conv_output_channels * timesteps * 4 +     # Conv output
            self.lstm_hidden_size * timesteps * 4 +         # LSTM output
            self.lstm_hidden_size * self.lstm_num_layers * 2 * 4 +  # LSTM states (h, c)
            9 * timesteps * 4                               # Output
        )

        total_runtime = activation_mem + model_size_int8 + 30000  # +30KB overhead

        return {
            'parameters': params,
            'model_size_fp32': params * 4,
            'model_size_int8': model_size_int8,
            'activation_memory': activation_mem,
            'estimated_total_runtime': total_runtime,
            'fits_nrf52840': total_runtime < 220000,
            'window_samples': window_samples,
            'output_timesteps': timesteps,
        }


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention for EMG signals.

    This module learns to weight channels based on their importance
    at each timestep, allowing the model to focus on the most relevant
    EMG channels for the current gesture.

    Parameters
    ----------
    channels : int
        Number of input channels
    reduction : int
        Reduction ratio for the bottleneck. Default: 4
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, channels, time)

        Returns
        -------
        torch.Tensor
            Attention-weighted tensor, same shape as input
        """
        b, c, _ = x.shape
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)  # (B, C)
        # Excitation: learn channel weights
        y = self.fc(y).view(b, c, 1)     # (B, C, 1)
        # Scale: apply attention weights
        return x * y.expand_as(x)


class M1_4Channel_Desktop(nn.Module):
    """
    Full-Capacity 4-Channel CNN+LSTM for Desktop Inference (Variant 1 - Scaled).

    This architecture scales up the 4-channel TinyML model to match the
    parameter count and capacity of the 7-channel M1 model (~410K params),
    optimized for desktop/laptop inference without hardware constraints.

    Design Philosophy:
    - Compensate for fewer input channels (4 vs 7) with wider conv filters (144 vs 128)
    - Maintain full LSTM capacity (128 hidden, 3 layers) for temporal modeling
    - No memory constraints - prioritize accuracy over size
    - Target: 99%+ validation accuracy on 4-channel subset

    Architecture:
        Conv1d(4, 144, k=15, s=10) → BatchNorm → ReLU → Dropout(0.2)
        → LSTM(144, 128, num_layers=3) → LayerNorm → FC(9)

    Parameters: ~405K (matches 7-channel M1)
    Expected Accuracy: 99.3-99.6%
    Inference Time: 10-12ms CPU, 2-4ms GPU

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 4
    conv_output_channels : int
        Number of convolutional output channels. Default: 144
    kernel_width : int
        Convolutional kernel width. Default: 15 (7.5ms at 2kHz)
    stride : int
        Convolutional stride. Default: 10 (downsample 2kHz→200Hz)
    lstm_hidden_size : int
        LSTM hidden state size. Default: 128
    lstm_num_layers : int
        Number of stacked LSTM layers. Default: 3
    output_channels : int
        Number of output gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.2
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv_output_channels: int = 144,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_output_channels = conv_output_channels

        # Temporal alignment parameters
        self.left_context = kernel_width - 1
        self.stride = stride

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial Conv1D
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )
        self.conv_bn = nn.BatchNorm1d(conv_output_channels)
        self.conv_dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 3. Three-layer LSTM for hierarchical temporal patterns
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        # 4. Post-LSTM normalization
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 5. Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Conv layer
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # LSTM: Orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 4, time)

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, time')
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Conv1D with BatchNorm
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.relu(x)
        x = self.conv_dropout(x)

        # 3. Transpose for LSTM
        x = x.transpose(1, 2)  # (B, T', 144)

        # 4. LSTM (3 layers)
        x = x.contiguous()
        x, _ = self.lstm(x)  # (B, T', 128)

        # 5. Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # 6. Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # 7. Transpose back
        x = x.transpose(1, 2)  # (B, 9, T')

        return x


class M1_4Channel_Optimized(nn.Module):
    """
    Optimized 4-Channel CNN+BiLSTM for Desktop Inference (Variant 2 - Attention).

    This architecture is custom-designed for 4-channel EMG characteristics,
    using multi-scale feature extraction and channel attention to compensate
    for fewer input channels with richer per-channel analysis.

    Design Philosophy:
    - Multi-scale conv paths capture different EMG activation speeds
    - Channel attention dynamically weights channel importance
    - Bidirectional LSTM (desktop can look ahead)
    - Moderate depth to avoid overfitting on limited channels

    Architecture:
        Multi-scale Conv (4→64, kernels 7/15/25) → BatchNorm → Attention
        → Downsample Conv (64→96, s=10) → Residual Conv
        → BiLSTM(96, 48, 1 layer) → LayerNorm → FC(9)

    Parameters: ~135-155K (3× TinyML, 0.37× 7ch full)
    Expected Accuracy: 99.55-99.65%
    Inference Time: 15-25ms CPU (BiLSTM slower)

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 4
    multiscale_channels : list
        Output channels for each parallel conv path. Default: [20, 22, 22]
    multiscale_kernels : list
        Kernel sizes for parallel paths. Default: [7, 15, 25]
    attention_reduction : int
        Reduction ratio for channel attention. Default: 4
    conv_output_channels : int
        Channels after downsampling conv. Default: 96
    kernel_width : int
        Kernel for downsampling conv. Default: 15
    stride : int
        Stride for downsampling. Default: 10
    residual_kernel : int
        Kernel for residual conv (0 to disable). Default: 5
    lstm_hidden_size : int
        BiLSTM hidden size (×2 for output). Default: 48
    lstm_num_layers : int
        Number of BiLSTM layers. Default: 1
    output_channels : int
        Number of gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.3
    """

    def __init__(
        self,
        input_channels: int = 4,
        multiscale_channels: list = None,
        multiscale_kernels: list = None,
        attention_reduction: int = 4,
        conv_output_channels: int = 96,
        kernel_width: int = 15,
        stride: int = 10,
        residual_kernel: int = 5,
        lstm_hidden_size: int = 48,
        lstm_num_layers: int = 1,
        output_channels: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()

        if multiscale_channels is None:
            multiscale_channels = [20, 22, 22]
        if multiscale_kernels is None:
            multiscale_kernels = [7, 15, 25]

        self.input_channels = input_channels
        self.stride = stride
        total_multiscale_channels = sum(multiscale_channels)

        # Temporal alignment
        self.left_context = kernel_width - 1

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Multi-scale Conv Feature Extraction
        self.multiscale_convs = nn.ModuleList([
            nn.Conv1d(input_channels, out_ch, kernel_size=k, padding=k//2)
            for out_ch, k in zip(multiscale_channels, multiscale_kernels)
        ])
        self.multiscale_bn = nn.BatchNorm1d(total_multiscale_channels)
        self.multiscale_relu = nn.ReLU()
        self.multiscale_dropout = nn.Dropout(dropout * 0.67)  # 0.2 for 0.3

        # 3. Channel Attention
        self.channel_attention = ChannelAttention(
            channels=total_multiscale_channels,
            reduction=attention_reduction
        )

        # 4. Downsampling Conv
        self.conv = nn.Conv1d(
            total_multiscale_channels,
            conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )
        self.conv_bn = nn.BatchNorm1d(conv_output_channels)
        self.conv_relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout)

        # 5. Optional Residual Conv Block
        if residual_kernel > 0:
            self.residual_conv = nn.Conv1d(
                conv_output_channels,
                conv_output_channels,
                kernel_size=residual_kernel,
                padding=residual_kernel // 2,
            )
            self.residual_bn = nn.BatchNorm1d(conv_output_channels)
            self.residual_relu = nn.ReLU()
        else:
            self.residual_conv = None

        # 6. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_size = lstm_hidden_size * 2  # Bidirectional
        self.lstm_ln = nn.LayerNorm(lstm_output_size)

        # 7. Output projection
        self.fc_out = nn.Linear(lstm_output_size, output_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias

        # Output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 4, time)

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, time')
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Multi-scale Conv Feature Extraction
        multiscale_features = [conv(x) for conv in self.multiscale_convs]
        x = torch.cat(multiscale_features, dim=1)  # (B, 64, T)
        x = self.multiscale_bn(x)
        x = self.multiscale_relu(x)
        x = self.multiscale_dropout(x)

        # 3. Channel Attention
        x = self.channel_attention(x)  # (B, 64, T)

        # 4. Downsampling Conv
        x = self.conv(x)       # (B, 96, T')
        x = self.conv_bn(x)
        x = self.conv_relu(x)
        x = self.conv_dropout(x)

        # 5. Optional Residual Conv
        if self.residual_conv is not None:
            identity = x
            x = self.residual_conv(x)
            x = self.residual_bn(x)
            x = self.residual_relu(x)
            x = x + identity  # Residual connection

        # 6. BiLSTM
        x = x.transpose(1, 2)  # (B, T', 96)
        x = x.contiguous()
        x, _ = self.lstm(x)    # (B, T', 96)
        x = self.lstm_ln(x)

        # 7. Output projection
        x = self.fc_out(x)     # (B, T', 9)
        x = x.transpose(1, 2)  # (B, 9, T')

        return x


class M1_4Channel_Efficient(nn.Module):
    """
    Efficient 4-Channel Pure CNN for Fast Desktop Inference (Variant 3 - Fast).

    This architecture replaces LSTM with dilated and depthwise separable
    convolutions for 5-10× faster inference on desktop CPUs while maintaining
    98-99% accuracy. Optimized for real-time applications (<10ms latency).

    Design Philosophy:
    - NO LSTM - eliminates sequential bottleneck
    - Dilated convolutions for long-range temporal context
    - Depthwise separable conv for efficient multi-scale features
    - Fully parallelizable across timesteps

    Architecture:
        Conv1d(4→48, k=15, s=10) → BatchNorm → ReLU → Dropout
        → DilatedConv(48→64, k=7, d=2) → BatchNorm → ReLU → Dropout
        → DepthwiseSep(64→96, k=11) → BatchNorm → ReLU → Dropout
        → Conv1d(96→128, k=5, s=2) → BatchNorm → ReLU
        → Conv1d(128→64, k=3) → BatchNorm → ReLU → Dropout
        → Conv1d(64→9, k=1)

    Parameters: ~120K
    Expected Accuracy: 98.0-99.0%
    Inference Time: 2-3ms CPU (5-10× faster than LSTM!)

    Parameters
    ----------
    input_channels : int
        Number of EMG input channels. Default: 4
    conv1_channels : int
        Initial conv output channels. Default: 48
    dilated_channels : int
        Dilated conv output channels. Default: 64
    depthwise_channels : int
        Depthwise separable output channels. Default: 96
    refined_channels : int
        Refinement conv channels. Default: 128
    head_channels : int
        Classification head channels. Default: 64
    output_channels : int
        Number of gesture classes. Default: 9
    dropout : float
        Dropout probability. Default: 0.2
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv1_channels: int = 48,
        dilated_channels: int = 64,
        depthwise_channels: int = 96,
        refined_channels: int = 128,
        head_channels: int = 64,
        output_channels: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        # Temporal alignment
        self.left_context = 7
        self.stride = 20  # Overall: 10 (conv1) × 2 (conv4)

        # 1. Amplitude normalization
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial temporal downsampling (2kHz → 200Hz)
        self.conv1 = nn.Conv1d(input_channels, conv1_channels,
                               kernel_size=15, stride=10, padding=7)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.dropout1 = nn.Dropout(dropout)

        # 3. Dilated conv for long-range dependencies (65ms receptive field)
        self.conv2_dilated = nn.Conv1d(conv1_channels, dilated_channels,
                                       kernel_size=7, dilation=2, padding=6)
        self.bn2 = nn.BatchNorm1d(dilated_channels)
        self.dropout2 = nn.Dropout(dropout)

        # 4. Depthwise separable conv (efficient multi-scale)
        self.conv3_depthwise = nn.Conv1d(dilated_channels, dilated_channels,
                                         kernel_size=11, groups=dilated_channels,
                                         padding=5)
        self.conv3_pointwise = nn.Conv1d(dilated_channels, depthwise_channels,
                                         kernel_size=1)
        self.bn3 = nn.BatchNorm1d(depthwise_channels)
        self.dropout3 = nn.Dropout(dropout)

        # 5. Temporal pooling + refinement (200Hz → 100Hz)
        self.conv4 = nn.Conv1d(depthwise_channels, refined_channels,
                              kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm1d(refined_channels)

        # 6. Classification head
        self.conv5 = nn.Conv1d(refined_channels, head_channels,
                              kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(head_channels)
        self.dropout5 = nn.Dropout(dropout + 0.1)  # Higher before output

        # 7. Output projection
        self.conv_out = nn.Conv1d(head_channels, output_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EMG, shape (batch, 4, 2000) - 1.0s @ 2kHz

        Returns
        -------
        torch.Tensor
            Gesture logits, shape (batch, 9, 100)
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Initial conv (2kHz → 200Hz)
        x = self.conv1(x)  # (B, 48, 200)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout1(x)

        # 3. Dilated conv (captures 65ms context)
        x = self.conv2_dilated(x)  # (B, 64, 200)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout2(x)

        # 4. Depthwise separable (efficient multi-scale)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)  # (B, 96, 200)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.dropout3(x)

        # 5. Temporal pooling (200Hz → 100Hz)
        x = self.conv4(x)  # (B, 128, 100)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)

        # 6. Classification head
        x = self.conv5(x)  # (B, 64, 100)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = self.dropout5(x)

        # 7. Output
        x = self.conv_out(x)  # (B, 9, 100)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
