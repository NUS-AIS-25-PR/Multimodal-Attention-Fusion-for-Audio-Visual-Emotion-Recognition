"""
Compute Table 6 metrics for the 4 model variants.
Outputs: total params, trainable params, GFLOPs, inference time (ms/sample).
Precision/Recall/Acc/F1 are computed separately from confusion matrices.

Run from project root:
    python src/benchmark_table6.py
"""
from __future__ import annotations

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.audio import AudioNet
from models.video import VideoNet
from models.fusion import FusionModel
from models.wavlm_audio import WavLMAudioEncoder


# ── Confusion matrices read from wandb charts ───────────────────────────────
# Rows = True label (0..7), Cols = Predicted label
# Classes: 0=neutral,1=calm,2=happy,3=sad,4=angry,5=fearful,6=disgust,7=surprised

CM = {
    "audio": np.array([
        [13,  2,  0,  0,  0,  0,  0,  0],
        [ 4, 25,  0,  1,  0,  0,  0,  0],
        [ 0,  0, 24,  0,  4,  0,  0,  2],
        [ 1,  3,  3, 17,  1,  3,  0,  2],
        [ 0,  0,  1,  0, 27,  0,  2,  0],
        [ 0,  0,  1,  1,  0, 27,  1,  0],
        [ 0,  0,  1,  2,  1,  0, 26,  0],
        [ 0,  0,  5,  0,  0,  1,  0, 24],
    ]),
    "video": np.array([
        [12,  2,  0,  0,  0,  0,  0,  1],
        [ 2, 23,  1,  1,  0,  0,  0,  3],
        [ 0,  0, 26,  0,  0,  0,  0,  4],
        [ 4,  0,  0, 19,  4,  0,  1,  2],
        [ 0,  0,  0,  1, 28,  0,  0,  1],
        [ 0,  0,  0,  2,  1, 15,  5,  7],
        [ 0,  0,  0,  0,  1,  0, 29,  0],
        [ 3,  0,  0,  0,  2,  2,  0, 23],
    ]),
    "gated": np.array([
        [15,  0,  0,  0,  0,  0,  0,  0],
        [ 1, 26,  1,  2,  0,  0,  0,  0],
        [ 0,  0, 29,  0,  0,  0,  0,  1],
        [ 2,  0,  0, 26,  1,  0,  0,  1],
        [ 0,  0,  0,  0, 29,  0,  0,  1],
        [ 0,  0,  0,  1,  0, 25,  0,  4],
        [ 0,  0,  0,  0,  0,  0, 30,  0],
        [ 0,  0,  0,  0,  0,  0,  0, 30],
    ]),
    "xattn": np.array([
        [10,  0,  0,  0,  0,  0,  0,  0],
        [ 0, 19,  0,  0,  0,  0,  0,  1],
        [ 0,  0, 20,  0,  0,  0,  0,  0],
        [ 1,  0,  0, 17,  1,  1,  0,  0],
        [ 0,  0,  0,  0, 19,  0,  0,  1],
        [ 0,  0,  0,  1,  2, 14,  0,  3],
        [ 0,  0,  0,  0,  0,  0, 19,  1],
        [ 0,  0,  0,  0,  0,  0,  0, 20],
    ]),
}


def cm_metrics(cm: np.ndarray) -> dict:
    """Compute macro precision, recall, F1, accuracy from confusion matrix."""
    n = cm.shape[0]
    total = cm.sum()
    acc = cm.diagonal().sum() / total

    prec, rec, f1 = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp   # others predicted as i
        fn = cm[i, :].sum() - tp   # i predicted as others
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(f)

    return {
        "acc":  acc,
        "prec": float(np.mean(prec)),
        "rec":  float(np.mean(rec)),
        "f1":   float(np.mean(f1)),
    }


def count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model: nn.Module, dummy_inputs: tuple, device: torch.device) -> float:
    """Return GFLOPs using torchinfo if available, else return -1."""
    try:
        from torchinfo import summary
        inputs = tuple(x.to(device) for x in dummy_inputs)
        s = summary(model, input_data=inputs, verbose=0, device=device)
        return s.total_mult_adds / 1e9
    except Exception:
        return -1.0


@torch.no_grad()
def measure_inference_ms(
    model: nn.Module,
    dummy_inputs: tuple,
    device: torch.device,
    warmup: int = 20,
    runs: int = 100,
) -> float:
    """Return mean per-sample inference time in milliseconds."""
    model.eval()
    inputs = tuple(x.to(device) for x in dummy_inputs)

    # Warm-up
    for _ in range(warmup):
        model(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / runs * 1000  # ms per sample


def build_audio(device: torch.device) -> nn.Module:
    """WavLMAudioEncoder — stage-2 (last 2 layers unfrozen)."""
    m = WavLMAudioEncoder(num_classes=8, temporal_pooling="mean")
    m.unfreeze_backbone(num_last_layers=2)
    return m.to(device)


def build_video(device: torch.device) -> nn.Module:
    return VideoNet(num_classes=8, temporal_pooling="mean").to(device)


def build_gated(device: torch.device) -> nn.Module:
    """Gated fusion — stage-2 freeze policy mirrored from training."""
    audio = WavLMAudioEncoder(num_classes=8, temporal_pooling="mean")
    video = VideoNet(num_classes=8, temporal_pooling="mean")
    model = FusionModel(audio, video, num_classes=8, mode="gated", common_dim=256)
    # Reproduce stage-2 trainability: last 2 WavLM layers + last ResNet block + fusion head
    for p in model.audio_model.parameters():
        p.requires_grad = False
    model.audio_model.unfreeze_backbone(2)
    for p in model.audio_model.classifier.parameters():
        p.requires_grad = True
    for p in model.video_model.parameters():
        p.requires_grad = False
    for p in model.video_model.backbone[-1].parameters():   # layer4
        p.requires_grad = True
    for p in model.video_model.classifier.parameters():
        p.requires_grad = True
    return model.to(device)


def build_xattn(device: torch.device) -> nn.Module:
    """Cross-attention fusion (xattn_head=gated, d_model=96, heads=4) — stage-2."""
    audio = WavLMAudioEncoder(num_classes=8, temporal_pooling="mean")
    video = VideoNet(num_classes=8, temporal_pooling="mean")
    model = FusionModel(
        audio, video,
        num_classes=8,
        mode="xattn",
        xattn_head="gated",
        d_model=96,
        num_heads=4,
        xattn_attn_dropout=0.1,
        xattn_stochastic_depth=0.1,
        temporal_pooling="mean",
    )
    for p in model.audio_model.parameters():
        p.requires_grad = False
    model.audio_model.unfreeze_backbone(2)
    for p in model.audio_model.classifier.parameters():
        p.requires_grad = True
    for p in model.video_model.parameters():
        p.requires_grad = False
    for p in model.video_model.backbone[-1].parameters():
        p.requires_grad = True
    for p in model.video_model.classifier.parameters():
        p.requires_grad = True
    return model.to(device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Dummy inputs: batch=1, single sample
    # Audio: raw waveform [1, 48000] (3 s @ 16 kHz)
    # Video: [1, 8, 3, 112, 112]
    wav   = torch.randn(1, 48000)
    video = torch.randn(1, 8, 3, 112, 112)

    configs = {
        "Audio Only":  (build_audio,  (wav,)),
        "Video Only":  (build_video,  (video,)),
        "Gated Fusion":(build_gated,  (video, wav)),
        "Cross-Attn":  (build_xattn,  (video, wav)),
    }
    cm_keys = ["audio", "video", "gated", "xattn"]

    header = (
        f"{'Model':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'Total(M)':>10} {'Train(M)':>10} {'GFLOPs':>8} {'Time(ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for (name, (builder, dummy)), ck in zip(configs.items(), cm_keys):
        m = builder(device)
        m.eval()

        total_p, train_p = count_params(m)
        gflops = estimate_flops(m, dummy, device)
        ms     = measure_inference_ms(m, dummy, device)
        met    = cm_metrics(CM[ck])

        gf_str = f"{gflops:.2f}" if gflops >= 0 else "N/A"
        print(
            f"{name:<18} {met['acc']:>6.4f} {met['prec']:>6.4f} {met['rec']:>6.4f} "
            f"{met['f1']:>6.4f} {total_p/1e6:>10.2f} {train_p/1e6:>10.2f} "
            f"{gf_str:>8} {ms:>10.1f}"
        )

        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nNote: FLOPs require 'pip install torchinfo'. Time = mean over 100 runs (1 sample, no batching).")


if __name__ == "__main__":
    main()
