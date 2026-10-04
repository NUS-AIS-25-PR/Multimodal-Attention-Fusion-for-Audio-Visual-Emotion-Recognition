"""Adapted ICPR 2022 Intermediate Attention, using shared project encoders.

Reference: katerynaCh/multimodal-emotion-recognition, commit 65232ce,
MultiModalCNN.forward_feature_2. This reimplementation preserves one-head
query-summed attention modulation, not the original MFCC/EfficientFace stack.
"""
from __future__ import annotations

import torch
from torch import nn


class ChumachenkoIntermediateAttentionFusion(nn.Module):
    def __init__(self, audio_model: nn.Module, video_model: nn.Module,
                 num_classes: int = 8, d_model: int = 128):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.num_heads = 1
        self.audio_proj = nn.Linear(audio_model.sequence_dim, d_model)
        self.video_proj = nn.Linear(video_model.embedding_dim, d_model)
        self.audio_query = nn.Linear(d_model, d_model, bias=False)
        self.video_key = nn.Linear(d_model, d_model, bias=False)
        self.video_query = nn.Linear(d_model, d_model, bias=False)
        self.audio_key = nn.Linear(d_model, d_model, bias=False)
        self.classifier = nn.Linear(2 * d_model, num_classes)
        self.scale = d_model ** -0.5

    def modulate(self, audio: torch.Tensor, video: torch.Tensor):
        # [B, Ta, Tv] and [B, Tv, Ta]; softmax over keys, sum over queries.
        av = (self.audio_query(audio) @ self.video_key(video).transpose(1, 2)) * self.scale
        va = (self.video_query(video) @ self.audio_key(audio).transpose(1, 2)) * self.scale
        video_weight = av.softmax(dim=-1).sum(dim=1).unsqueeze(-1)
        audio_weight = va.softmax(dim=-1).sum(dim=1).unsqueeze(-1)
        return audio * audio_weight, video * video_weight

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        a = self.audio_proj(self.audio_model.encode_sequence(audio))
        b, t, c, h, w = video.shape
        v = self.video_model.backbone(video.reshape(b * t, c, h, w))
        v = self.video_proj(v.reshape(b, t, self.video_model.embedding_dim))
        a, v = self.modulate(a, v)
        return self.classifier(torch.cat([a.mean(dim=1), v.mean(dim=1)], dim=-1))
