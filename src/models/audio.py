"""
Mel-spectrogram audio encoder (legacy branch).

NOTE: This module was an early experimental audio pipeline based on mel-spectrogram
features. It was explored during development but was NOT used in the final paper
experiments. The paper uses WavLMAudioEncoder (wavlm_audio.py) as the audio encoder.

This file is retained as a reference implementation. The mel-spectrogram path is
still reachable in FusionModel via the audio_time_conv fallback when the audio
encoder does not expose an encode_sequence() interface.
"""
from __future__ import annotations

import torch
from torch import nn
import torchaudio.transforms as T

from .temporal import TemporalPooler


class SpecAugment(nn.Module):
    """SpecAugment: frequency and time masking augmentation for mel-spectrograms.

    NOTE: Only active when AudioNet (mel branch) is used. Not applied in
    WavLM-based experiments reported in the paper.
    """
    def __init__(self, freq_mask_param: int = 20, time_mask_param: int = 40, num_masks: int = 2, p: float = 0.5):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: spectrogram tensor [B, 1, n_mels, T] or [B, n_mels, T]
        Returns:
            augmented spectrogram
        """
        if not self.training or torch.rand(1).item() > self.p:
            return x
        
        # Handle both [B, 1, n_mels, T] and [B, n_mels, T]
        if x.dim() == 4:
            # [B, 1, n_mels, T]
            b, c, n_mels, time_steps = x.shape
            x = x.squeeze(1)  # [B, n_mels, T]
        else:
            b, n_mels, time_steps = x.shape
        
        for _ in range(self.num_masks):
            # Frequency masking
            if self.freq_mask_param > 0:
                freq_mask_len = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
                if freq_mask_len > 0:
                    freq_mask_start = torch.randint(0, n_mels - freq_mask_len, (1,)).item()
                    x[:, freq_mask_start:freq_mask_start + freq_mask_len, :] = 0
            
            # Time masking
            if self.time_mask_param > 0:
                time_mask_len = torch.randint(0, self.time_mask_param + 1, (1,)).item()
                if time_mask_len > 0:
                    time_mask_start = torch.randint(0, time_steps - time_mask_len, (1,)).item()
                    x[:, :, time_mask_start:time_mask_start + time_mask_len] = 0
        
        return x.unsqueeze(1)  # Add channel dimension back: [B, 1, n_mels, T]


class AudioResNet18(nn.Module):
    """ResNet18 adapted for mel-spectrogram input (single channel).

    NOTE: Legacy mel branch — not used in paper experiments.
    """
    def __init__(self, embedding_dim: int = 128, temporal_bins: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temporal_bins = temporal_bins
        
        # Use ResNet18 architecture but adapted for 1-channel input
        # Initial conv: 1 channel -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, temporal_bins))
        self.fc = nn.Linear(512, embedding_dim)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        # Downsample layer if needed
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
        
        # Basic blocks
        for _ in range(blocks):
            layers.append(self._make_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.sequence_pool(x).squeeze(2).transpose(1, 2).contiguous()
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward_sequence(x)
        return seq.mean(dim=1)


class AudioCNN(nn.Module):
    """Lightweight CNN for mel-spectrogram input.

    NOTE: Legacy mel branch — not used in paper experiments.
    """
    def __init__(self, embedding_dim: int = 128, temporal_bins: int = 16) -> None:
        super().__init__()
        self.temporal_bins = temporal_bins
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, temporal_bins))
        self.embedding_dim = embedding_dim

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.sequence_pool(x).squeeze(2).transpose(1, 2).contiguous()
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.forward_sequence(x)
        return seq.mean(dim=1)


class AudioNet(nn.Module):
    """Mel-spectrogram audio encoder wrapping AudioResNet18 or AudioCNN.

    NOTE: Legacy mel branch — not used in paper experiments. The paper uses
    WavLMAudioEncoder from wavlm_audio.py as the audio encoder.
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        use_resnet: bool = True,
        spec_augment: bool = True,
        temporal_pooling: str = "mean",
        temporal_num_heads: int = 4,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_dim = embedding_dim
        
        # Use ResNet18 or lightweight CNN as encoder
        if use_resnet:
            self.encoder = AudioResNet18(embedding_dim=embedding_dim)
        else:
            self.encoder = AudioCNN(embedding_dim=embedding_dim)
        self.temporal_pool = TemporalPooler(
            dim=embedding_dim,
            mode=temporal_pooling,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
        )
        
        # SpecAugment for training
        self.spec_augment = SpecAugment(freq_mask_param=20, time_mask_param=40, p=0.5) if spec_augment else None
        
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SpecAugment during training
        if self.spec_augment is not None and self.training:
            x = self.spec_augment(x)
        seq = self.encoder.forward_sequence(x)
        return self.temporal_pool(seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encode(x)
        return self.classifier(emb)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder-derived temporal states for fusion models."""
        if self.spec_augment is not None and self.training:
            x = self.spec_augment(x)
        return self.encoder.forward_sequence(x)
