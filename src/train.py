from __future__ import annotations

import os
import argparse
import platform
import sys
import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows encoding issue with wandb
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data.ravdess import DATASET_FACTORY, PAIR_SERVICE, SPLIT_SERVICE
from models.audio import AudioNet
from models.video import VideoNet
from models.fusion import FusionModel
from models.chumachenko_ia import ChumachenkoIntermediateAttentionFusion
from models.wavlm_audio import WavLMAudioEncoder
from utils.metrics import accuracy, classification_metrics
from utils.seed import set_seed


def parse_actor_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x) for x in text.split(",")]


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def _build_loader_kwargs(
    data_root: Path,
    device: torch.device,
    requested_num_workers: int,
) -> Dict[str, object]:
    if requested_num_workers >= 0:
        num_workers = requested_num_workers
    else:
        data_root_resolved = data_root.expanduser().resolve()
        is_mnt = str(data_root_resolved).startswith("/mnt/")
        if sys.platform == "win32":
            num_workers = 0
        elif _is_wsl() and is_mnt:
            # WSL + Windows mount is typically slower and less stable for heavy multiprocessing I/O.
            num_workers = 0
        elif _is_wsl():
            num_workers = 2
        else:
            cpu_count = os.cpu_count() or 4
            num_workers = min(8, max(2, cpu_count // 2))

    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_dataloaders(
    data_root: Path,
    num_classes: int,
    num_frames: int,
    train_actors: List[int],
    val_actors: List[int],
    test_actors: List[int],
    batch_size: int,
    device: torch.device,
    requested_num_workers: int = -1,
    stratified: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_wavlm: bool = False,
    train_augment: bool = True,
    use_face_crop: bool = True,
    pairs_csv: Path = Path("pairs.csv"),
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    pairs = PAIR_SERVICE.build_pairs(data_root)
    PAIR_SERVICE.save_pairs_csv(pairs, pairs_csv)

    if len(pairs) == 0:
        raise RuntimeError("No audio-video pairs found. Check data_root and filenames.")

    print(f"Total pairs: {len(pairs)}")
    for p in pairs[:3]:
        print(
            f"Sample pair: actor={p.actor} emotion={p.emotion} intensity={p.intensity} "
            f"statement={p.statement} repetition={p.repetition}"
        )

    # Use stratified split or actor-based split
    if stratified:
        print("Using stratified split (balanced emotion distribution)")
        test_ratio = max(0.0, 1.0 - float(train_ratio) - float(val_ratio))
        train_pairs, val_pairs, test_pairs = SPLIT_SERVICE.stratified(
            pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42,
        )
    else:
        print("Using actor-based split")
        train_pairs, val_pairs, test_pairs = SPLIT_SERVICE.by_actor(
            pairs, train_actors, val_actors, test_actors
        )
    
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs)}")

    def class_dist(ps: List) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for p in ps:
            label = PAIR_SERVICE.map_emotion_label(p.emotion, num_classes)
            counts[label] = counts.get(label, 0) + 1
        return counts

    print(f"Train class distribution: {class_dist(train_pairs)}")
    print(f"Val class distribution: {class_dist(val_pairs)}")
    print(f"Test class distribution: {class_dist(test_pairs)}")

    # Apply augmentation only to training dataset, face crop to all
    train_ds = DATASET_FACTORY.create(
        train_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=train_augment,
        use_face_crop=use_face_crop,
        use_wavlm=use_wavlm,
    )
    val_ds = DATASET_FACTORY.create(
        val_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=False,
        use_face_crop=use_face_crop,
        use_wavlm=use_wavlm,
    )
    test_ds = DATASET_FACTORY.create(
        test_pairs,
        num_classes=num_classes,
        num_frames=num_frames,
        augment=False,
        use_face_crop=use_face_crop,
        use_wavlm=use_wavlm,
    )

    loader_kwargs = _build_loader_kwargs(
        data_root=data_root,
        device=device,
        requested_num_workers=requested_num_workers,
    )
    print(
        "DataLoader config: "
        f"num_workers={loader_kwargs['num_workers']}, "
        f"pin_memory={loader_kwargs['pin_memory']}, "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    sizes = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    return train_loader, val_loader, test_loader, sizes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    fusion_mode: str,
    fusion_align_weight: float = 0.0,
) -> Dict[str, float]:
    model.train()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    for video, audio, labels, _ in tqdm(loader, desc="train", leave=False):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # For single-modality models (audio/video), pass only that modality
        # For fusion models, pass both
        if fusion_mode in {"audio", "video"}:
            outputs = model(audio if fusion_mode == "audio" else video)
        else:
            outputs = model(video, audio)
        if fusion_mode == "late":
            log_probs = torch.log(outputs + 1e-8)
            cls_loss = loss_fn(log_probs, labels)
            contrastive_loss_value = cls_loss.new_zeros(())
            preds = outputs.argmax(dim=1)
        else:
            cls_loss = loss_fn(outputs, labels)
            contrastive_loss_value = cls_loss.new_zeros(())
            preds = outputs.argmax(dim=1)
            if fusion_align_weight > 0.0 and hasattr(model, "pop_alignment_loss"):
                align_loss = model.pop_alignment_loss()
                if align_loss is not None:
                    contrastive_loss_value = align_loss
        loss = cls_loss + fusion_align_weight * contrastive_loss_value

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_cls_loss += cls_loss.item() * labels.size(0)
        total_contrastive_loss += contrastive_loss_value.item() * labels.size(0)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    scores = classification_metrics(all_preds, all_targets, num_classes=outputs.shape[-1])
    return {
        "loss": total_loss / len(loader.dataset),
        "cls_loss": total_cls_loss / len(loader.dataset),
        "contrastive_loss": total_contrastive_loss / len(loader.dataset),
        "acc": accuracy(all_preds, all_targets),
        "f1": scores["macro_f1"],
        **scores,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    fusion_mode: str,
    fusion_align_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    for video, audio, labels, _ in tqdm(loader, desc="eval", leave=False):
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        # For single-modality models (audio/video), pass only that modality
        # For fusion models, pass both
        if fusion_mode in {"audio", "video"}:
            outputs = model(audio if fusion_mode == "audio" else video)
        else:
            outputs = model(video, audio)
        if fusion_mode == "late":
            log_probs = torch.log(outputs + 1e-8)
            cls_loss = loss_fn(log_probs, labels)
            contrastive_loss_value = cls_loss.new_zeros(())
            preds = outputs.argmax(dim=1)
        else:
            cls_loss = loss_fn(outputs, labels)
            contrastive_loss_value = cls_loss.new_zeros(())
            preds = outputs.argmax(dim=1)
            if fusion_align_weight > 0.0 and hasattr(model, "pop_alignment_loss"):
                align_loss = model.pop_alignment_loss()
                if align_loss is not None:
                    contrastive_loss_value = align_loss
        loss = cls_loss + fusion_align_weight * contrastive_loss_value

        total_loss += loss.item() * labels.size(0)
        total_cls_loss += cls_loss.item() * labels.size(0)
        total_contrastive_loss += contrastive_loss_value.item() * labels.size(0)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    scores = classification_metrics(all_preds, all_targets, num_classes=outputs.shape[-1])
    return {
        "loss": total_loss / len(loader.dataset),
        "cls_loss": total_cls_loss / len(loader.dataset),
        "contrastive_loss": total_contrastive_loss / len(loader.dataset),
        "acc": accuracy(all_preds, all_targets),
        "f1": scores["macro_f1"],
        **scores,
    }


def plot_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> np.ndarray:
    """Generate confusion matrix plot."""
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "black")
    
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig


def build_model(
    num_classes: int,
    fusion: str,
    pretrained_video: bool = True,
    xattn_head: str = "concat",
    xattn_d_model: int = 128,
    xattn_heads: int = 4,
    xattn_attn_dropout: float = 0.1,
    xattn_stochastic_depth: float = 0.1,
    temporal_pooling: str = "mean",
    temporal_num_heads: int = 4,
    temporal_num_layers: int = 1,
    temporal_dropout: float = 0.1,
    audio_n_mels: int = 64,
    use_resnet_audio: bool = True,
    use_wavlm: bool = False,
    fusion_align_mode: str = "none",
    fusion_align_dim: int = 256,
    fusion_align_temperature: float = 0.07,
    xattn_use_emotion_prior: bool = False,
    xattn_emotion_prior_dim: int = 8,
    xattn_emotion_prior_hidden_dim: int = 64,
    xattn_emotion_prior_dropout: float = 0.1,
    smoke: bool = False,
    checkpoint_init: bool = False,
) -> nn.Module:
    if fusion == "audio":
        if use_wavlm:
            return WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
                smoke=smoke,
                checkpoint_init=checkpoint_init,
            )
        else:
            return AudioNet(
                num_classes=num_classes,
                use_resnet=use_resnet_audio,
                spec_augment=True,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
    if fusion == "video":
        return VideoNet(
            num_classes=num_classes,
            pretrained=pretrained_video,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
    if fusion in {"late", "concat", "gated", "chumachenko_ia"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
                smoke=smoke,
                checkpoint_init=checkpoint_init,
            )
        else:
            audio = AudioNet(
                num_classes=num_classes,
                use_resnet=use_resnet_audio,
                spec_augment=True,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        video = VideoNet(
            num_classes=num_classes,
            pretrained=pretrained_video,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
        if fusion == "chumachenko_ia":
            if not use_wavlm:
                raise ValueError("chumachenko_ia requires WavLM sequence features")
            return ChumachenkoIntermediateAttentionFusion(audio, video, num_classes)
        return FusionModel(
            audio,
            video,
            num_classes=num_classes,
            mode=fusion,
            fusion_align_mode=fusion_align_mode,
            fusion_align_dim=fusion_align_dim,
            fusion_align_temperature=fusion_align_temperature,
        )
    if fusion in {"xattn", "xattn_concat", "xattn_gated"}:
        if use_wavlm:
            audio = WavLMAudioEncoder(
                num_classes=num_classes,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
                smoke=smoke,
                checkpoint_init=checkpoint_init,
            )
        else:
            audio = AudioNet(
                num_classes=num_classes,
                use_resnet=use_resnet_audio,
                spec_augment=True,
                temporal_pooling=temporal_pooling,
                temporal_num_heads=temporal_num_heads,
                temporal_num_layers=temporal_num_layers,
                temporal_dropout=temporal_dropout,
            )
        video = VideoNet(
            num_classes=num_classes,
            pretrained=pretrained_video,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
        # map alias fusion names
        head = xattn_head
        if fusion == "xattn_concat":
            head = "concat"
        if fusion == "xattn_gated":
            head = "gated"
        return FusionModel(
            audio,
            video,
            num_classes=num_classes,
            mode="xattn",
            xattn_head=head,
            d_model=xattn_d_model,
            num_heads=xattn_heads,
            audio_n_mels=audio_n_mels if not use_wavlm else 768,  # WavLM has 768 hidden dim
            xattn_attn_dropout=xattn_attn_dropout,
            xattn_stochastic_depth=xattn_stochastic_depth,
            temporal_pooling=temporal_pooling,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
        )
    raise ValueError(f"Unknown fusion mode: {fusion}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=8, choices=[4, 8])
    parser.add_argument(
        "--fusion",
        type=str,
        default="audio",
        choices=["audio", "video", "late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated", "chumachenko_ia"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="stratified",
        choices=["actor", "stratified"],
        help="Data split mode: 'actor' (by actors) or 'stratified' (balanced emotion distribution)",
    )
    parser.add_argument("--train_actors", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18", help="Used only with --split_mode actor")
    parser.add_argument("--val_actors", type=str, default="19,20,21", help="Used only with --split_mode actor")
    parser.add_argument("--test_actors", type=str, default="22,23,24", help="Used only with --split_mode actor")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio (used only with --split_mode stratified)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio (used only with --split_mode stratified)")
    parser.add_argument("--no_pretrained_video", action="store_true")
    parser.add_argument("--use_cosine_annealing", action="store_true", help="Use cosine annealing learning rate scheduler")
    parser.add_argument(
        "--cosine_stage2_only",
        action="store_true",
        help="For two-stage fusion training: keep stage-1 LR fixed, apply cosine scheduler only in stage-2.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--xattn_head", type=str, choices=["concat", "gated"], default="concat", help="xattn fusion head when --fusion xattn is used")
    parser.add_argument("--xattn_d_model", type=int, default=128, help="d_model for cross-attention (default 128)")
    parser.add_argument("--xattn_heads", type=int, default=4, help="number of attention heads for xattn (default 4)")
    parser.add_argument(
        "--xattn_attn_dropout",
        type=float,
        default=0.1,
        help="Attention dropout probability for xattn MultiHeadAttention (default 0.1)",
    )
    parser.add_argument(
        "--xattn_stochastic_depth",
        type=float,
        default=0.1,
        help="Drop-path probability on xattn residual branches (default 0.1)",
    )
    parser.add_argument(
        "--xattn_use_emotion_prior",
        action="store_true",
        help="Enable emotion-prior-conditioned attention bias in xattn.",
    )
    parser.add_argument(
        "--xattn_emotion_prior_dim",
        type=int,
        default=8,
        help="Dimension of the global emotion prior vector used to bias xattn.",
    )
    parser.add_argument(
        "--xattn_emotion_prior_hidden_dim",
        type=int,
        default=64,
        help="Hidden size of the emotion prior MLP.",
    )
    parser.add_argument(
        "--xattn_emotion_prior_dropout",
        type=float,
        default=0.1,
        help="Dropout inside the emotion prior module.",
    )
    parser.add_argument(
        "--temporal_pooling",
        type=str,
        default="mean",
        choices=["mean", "attn", "transformer"],
        help="Temporal aggregation for sequence features: mean, learnable attention, or transformer.",
    )
    parser.add_argument(
        "--temporal_num_heads",
        type=int,
        default=4,
        help="Attention heads used by temporal transformer pooling (default 4).",
    )
    parser.add_argument(
        "--temporal_num_layers",
        type=int,
        default=1,
        help="Transformer layers used by temporal transformer pooling (default 1).",
    )
    parser.add_argument(
        "--temporal_dropout",
        type=float,
        default=0.1,
        help="Dropout used inside temporal attention/transformer pooling (default 0.1).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropyLoss in non-late modes (default 0.0)",
    )
    parser.add_argument("--audio_n_mels", type=int, default=64, help="n_mels used in audio preprocessing (default 64)")
    parser.add_argument(
        "--fusion_align_mode",
        type=str,
        default="none",
        choices=["none", "clip"],
        help="Optional semantic alignment before concat/gated fusion. 'clip' enables CLIP-style shared-space alignment.",
    )
    parser.add_argument(
        "--fusion_align_dim",
        type=int,
        default=256,
        help="Shared embedding dimension used by semantic alignment before concat/gated fusion.",
    )
    parser.add_argument(
        "--fusion_align_temperature",
        type=float,
        default=0.07,
        help="Initial temperature for CLIP-style alignment logits.",
    )
    parser.add_argument(
        "--fusion_align_weight",
        type=float,
        default=0.1,
        help="Weight for the auxiliary CLIP-style alignment loss when semantic alignment is enabled.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay (default: 1e-4)")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience in epochs (set 0 to disable)")
    parser.add_argument("--use_resnet_audio", action="store_true", help="Use ResNet18 for audio encoder (default: lightweight CNN)")
    parser.add_argument("--two_stage_training", action="store_true", help="Use two-stage training (freeze encoders, train fusion head first)")
    parser.add_argument("--use_wavlm", action="store_true", help="Use WavLM pretrained model for audio encoder")
    parser.add_argument("--wavlm_stage", type=int, default=1, choices=[1, 2], help="WavLM training stage: 1 (freeze backbone) or 2 (unfreeze last 2-4 layers)")
    parser.add_argument("--backbone_lr", type=float, default=3e-5, help="Learning rate for WavLM backbone in stage 2 (default: 3e-5)")
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=5,
        help="Epochs for stage-1 when --two_stage_training is enabled (encoder frozen, train fusion head only).",
    )
    parser.add_argument(
        "--audio_backbone_lr",
        type=float,
        default=1e-5,
        help="Stage-2 LR for audio encoder parameters in fusion training.",
    )
    parser.add_argument(
        "--video_backbone_lr",
        type=float,
        default=1e-5,
        help="Stage-2 LR for video encoder parameters in fusion training.",
    )
    parser.add_argument(
        "--fusion_unfreeze_wavlm_layers",
        type=int,
        default=2,
        help="Stage-2: unfreeze last N WavLM encoder layers for fusion training.",
    )
    parser.add_argument(
        "--fusion_unfreeze_video_blocks",
        type=int,
        default=1,
        help="Stage-2: unfreeze last N parameterized blocks in video backbone for fusion training.",
    )
    parser.add_argument(
        "--fusion_unfreeze_audio",
        action="store_true",
        default=True,
        help="Stage-2: unfreeze non-WavLM audio encoder parameters in fusion training (default: True).",
    )
    parser.add_argument(
        "--no_fusion_unfreeze_audio",
        dest="fusion_unfreeze_audio",
        action="store_false",
        help="Stage-2: keep non-WavLM audio encoder frozen in fusion training.",
    )
    parser.add_argument(
        "--audio_ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to initialize fusion audio branch (expects standalone audio model state_dict).",
    )
    parser.add_argument(
        "--video_ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to initialize fusion video branch (expects standalone video model state_dict).",
    )
    parser.add_argument("--use_face_crop", action="store_true", default=True, help="Use face detection and cropping (default: True)")
    parser.add_argument("--no_face_crop", dest="use_face_crop", action="store_false", help="Disable face detection and cropping")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1: auto for OS/WSL, 0: single-process, >0: fixed workers)",
    )
    return parser


class EmotionTrainer:
    """Object-oriented trainer orchestration preserving existing training behavior."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _init_tracking(self) -> None:
        if self.args.wandb:
            wandb.init(
                project="multimodal-emotion-recognition",
                name=f"{self.args.fusion}_epochs{self.args.epochs}_bs{self.args.batch_size}_{self.args.split_mode}",
                config={
                    "fusion": self.args.fusion,
                    "epochs": self.args.epochs,
                    "batch_size": self.args.batch_size,
                    "lr": self.args.lr,
                    "num_classes": self.args.num_classes,
                    "split_mode": self.args.split_mode,
                    "cosine_annealing": self.args.use_cosine_annealing,
                },
            )

    def _checkpoint_config(self) -> Dict[str, object]:
        return {
            **vars(self.args),
            "fusion": self.args.fusion,
            "use_wavlm": bool(self.args.use_wavlm),
            "xattn_head": getattr(self.args, "xattn_head", "concat"),
            "xattn_d_model": getattr(self.args, "xattn_d_model", 128),
            "xattn_heads": getattr(self.args, "xattn_heads", 4),
            "xattn_attn_dropout": getattr(self.args, "xattn_attn_dropout", 0.1),
            "xattn_stochastic_depth": getattr(self.args, "xattn_stochastic_depth", 0.1),
            "xattn_use_emotion_prior": getattr(self.args, "xattn_use_emotion_prior", False),
            "xattn_emotion_prior_dim": getattr(self.args, "xattn_emotion_prior_dim", 8),
            "xattn_emotion_prior_hidden_dim": getattr(self.args, "xattn_emotion_prior_hidden_dim", 64),
            "xattn_emotion_prior_dropout": getattr(self.args, "xattn_emotion_prior_dropout", 0.1),
            "temporal_pooling": getattr(self.args, "temporal_pooling", "mean"),
            "temporal_num_heads": getattr(self.args, "temporal_num_heads", 4),
            "temporal_num_layers": getattr(self.args, "temporal_num_layers", 1),
            "temporal_dropout": getattr(self.args, "temporal_dropout", 0.1),
            "audio_n_mels": getattr(self.args, "audio_n_mels", 64),
            "use_resnet_audio": getattr(self.args, "use_resnet_audio", True),
            "fusion_align_mode": getattr(self.args, "fusion_align_mode", "none"),
            "fusion_align_dim": getattr(self.args, "fusion_align_dim", 256),
            "fusion_align_temperature": getattr(self.args, "fusion_align_temperature", 0.07),
            "fusion_align_weight": getattr(self.args, "fusion_align_weight", 0.1),
        }

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = trainable

    def _is_two_stage_fusion_enabled(self, model: Optional[nn.Module] = None) -> bool:
        if not self.args.two_stage_training:
            return False
        if self.args.fusion in {"audio", "video"}:
            return False
        if model is None:
            return True
        return hasattr(model, "audio_model") and hasattr(model, "video_model")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer, epochs_in_stage: int):
        if not self.args.use_cosine_annealing:
            return None

        t_max = max(1, int(epochs_in_stage))
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        # Group-wise eta_min to avoid raising low-LR backbone groups.
        # Example: lr=8e-6 -> eta_min=8e-7 (instead of a global 1e-5 floor).
        eta_mins = [max(1e-8, lr * 0.1) for lr in base_lrs]
        print(
            "[INFO] Cosine scheduler (group-wise): "
            + ", ".join(
                f"group{i}: base_lr={base_lrs[i]:.2e}, eta_min={eta_mins[i]:.2e}"
                for i in range(len(base_lrs))
            )
        )

        lambdas = []
        for base_lr, eta_min in zip(base_lrs, eta_mins):
            if base_lr <= 0.0:
                lambdas.append(lambda _epoch: 1.0)
                continue

            def lr_lambda(epoch: int, _base=base_lr, _min=eta_min, _tmax=t_max) -> float:
                # Step scheduler at epoch end: use (epoch+1) so first .step() decays LR.
                t = min(epoch + 1, _tmax)
                cosine = 0.5 * (1.0 + math.cos(math.pi * t / _tmax))
                lr = _min + (_base - _min) * cosine
                return lr / _base

            lambdas.append(lr_lambda)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    @staticmethod
    def _optimizer_lr_text(optimizer: torch.optim.Optimizer) -> str:
        lrs = [group["lr"] for group in optimizer.param_groups]
        if len(lrs) == 1:
            return f"{lrs[0]:.2e}"
        return "[" + ", ".join(f"{lr:.2e}" for lr in lrs) + "]"

    def _set_video_backbone_trainable(self, video_model: nn.Module, unfreeze_blocks: int) -> None:
        # Freeze whole video branch first, then unfreeze tail blocks in backbone.
        self._set_module_trainable(video_model, False)

        if unfreeze_blocks <= 0:
            return

        backbone = getattr(video_model, "backbone", None)
        if not isinstance(backbone, nn.Sequential):
            self._set_module_trainable(video_model, True)
            return

        parameterized_modules: List[nn.Module] = [
            module for module in backbone if len(list(module.parameters())) > 0
        ]
        for module in parameterized_modules[-unfreeze_blocks:]:
            self._set_module_trainable(module, True)

        if hasattr(video_model, "classifier"):
            self._set_module_trainable(video_model.classifier, True)

    def _apply_two_stage_freeze_policy(self, model: nn.Module, stage: int) -> None:
        if not self._is_two_stage_fusion_enabled(model):
            return

        # Fusion head parameters are always trainable.
        for name, param in model.named_parameters():
            if name.startswith("audio_model.") or name.startswith("video_model."):
                continue
            param.requires_grad = True

        if stage == 1:
            # Stage-1: freeze both encoders, train fusion head only.
            self._set_module_trainable(model.audio_model, False)
            self._set_module_trainable(model.video_model, False)
            return

        if stage != 2:
            raise ValueError(f"Unsupported stage for two-stage training: {stage}")

        # Stage-2: selectively unfreeze audio/video encoders.
        audio_model = model.audio_model
        if isinstance(audio_model, WavLMAudioEncoder):
            self._set_module_trainable(audio_model, False)
            self._set_module_trainable(audio_model.classifier, True)
            audio_model.unfreeze_backbone(max(0, int(self.args.fusion_unfreeze_wavlm_layers)))
        else:
            self._set_module_trainable(audio_model, bool(self.args.fusion_unfreeze_audio))

        self._set_video_backbone_trainable(
            model.video_model,
            unfreeze_blocks=max(0, int(self.args.fusion_unfreeze_video_blocks)),
        )

    def _build_fusion_stage_optimizer(self, model: nn.Module, stage: int) -> torch.optim.Optimizer:
        args = self.args
        fusion_params: List[torch.nn.Parameter] = []
        audio_params: List[torch.nn.Parameter] = []
        video_params: List[torch.nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("audio_model."):
                audio_params.append(param)
            elif name.startswith("video_model."):
                video_params.append(param)
            else:
                fusion_params.append(param)

        param_groups: List[Dict[str, object]] = []
        if stage == 1:
            if not fusion_params:
                raise RuntimeError("Stage-1 expects fusion parameters, but none are trainable.")
            param_groups.append({"name": "fusion", "params": fusion_params, "lr": args.lr})
        elif stage == 2:
            if fusion_params:
                param_groups.append({"name": "fusion", "params": fusion_params, "lr": args.lr})
            if audio_params:
                param_groups.append({"name": "audio", "params": audio_params, "lr": args.audio_backbone_lr})
            if video_params:
                param_groups.append({"name": "video", "params": video_params, "lr": args.video_backbone_lr})
            if not param_groups:
                raise RuntimeError("Stage-2 expects trainable parameters, but none are trainable.")
        else:
            raise ValueError(f"Unsupported optimizer stage: {stage}")

        summary = ", ".join(
            f"{group['name']}={sum(p.numel() for p in group['params']):,}@{group['lr']:.2e}"
            for group in param_groups
        )
        print(f"[INFO] Optimizer groups (stage {stage}): {summary}")
        for group in param_groups:
            group.pop("name", None)

        return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    def _build_optimizer(self, model: nn.Module, stage: int = 0) -> torch.optim.Optimizer:
        args = self.args
        if self._is_two_stage_fusion_enabled(model) and stage in {1, 2}:
            return self._build_fusion_stage_optimizer(model, stage=stage)

        # Setup optimizer based on WavLM stage
        if args.use_wavlm and args.fusion == "audio":
            if args.wavlm_stage == 1:
                model.unfreeze_backbone(0)  # Keep backbone frozen
                params_to_train = model.get_stage1_params()
                optimizer = torch.optim.Adam(params_to_train, lr=args.lr, weight_decay=args.weight_decay)
                print(f"WavLM Stage 1: Training classifier head only (lr={args.lr})")
                return optimizer
            model.unfreeze_backbone(num_last_layers=2)
            param_groups = model.get_stage2_params()
            optimizer = torch.optim.Adam(
                [
                    {"params": param_groups["backbone"], "lr": args.backbone_lr},
                    {"params": param_groups["head"], "lr": args.lr},
                ],
                weight_decay=args.weight_decay,
            )
            print(f"WavLM Stage 2: Training with backbone lr={args.backbone_lr}, head lr={args.lr}")
            return optimizer

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for optimizer.")
        return torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    @staticmethod
    def _extract_state_dict(checkpoint_obj: object) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return checkpoint_obj["model"]
        if isinstance(checkpoint_obj, dict):
            if checkpoint_obj and all(isinstance(k, str) for k in checkpoint_obj.keys()):
                return checkpoint_obj  # raw state_dict
        raise RuntimeError("Checkpoint format not supported. Expected {'model': state_dict} or raw state_dict.")

    def _load_fusion_branch_checkpoints(self, model: nn.Module) -> None:
        args = self.args
        # Branch checkpoint init is meaningful only for multimodal fusion models.
        if args.fusion in {"audio", "video"}:
            return
        if not hasattr(model, "audio_model") or not hasattr(model, "video_model"):
            return

        if getattr(args, "revision", None):
            from revision.protocol import validate_warm_start
            for modality in ("audio", "video"):
                source = getattr(args, f"{modality}_ckpt")
                if source:
                    validate_warm_start(Path(source), args.revision, modality)

        if args.audio_ckpt:
            audio_ckpt = Path(args.audio_ckpt).expanduser()
            if not audio_ckpt.exists():
                raise FileNotFoundError(f"Audio checkpoint not found: {audio_ckpt}")
            audio_obj = torch.load(audio_ckpt, map_location="cpu")
            audio_state = self._extract_state_dict(audio_obj)
            missing, unexpected = model.audio_model.load_state_dict(audio_state, strict=bool(getattr(args, "revision", None)))
            print(
                f"[INFO] Loaded audio branch checkpoint: {audio_ckpt} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if unexpected:
                print(f"[WARNING] Audio unexpected keys (first 8): {unexpected[:8]}")

        if args.video_ckpt:
            video_ckpt = Path(args.video_ckpt).expanduser()
            if not video_ckpt.exists():
                raise FileNotFoundError(f"Video checkpoint not found: {video_ckpt}")
            video_obj = torch.load(video_ckpt, map_location="cpu")
            video_state = self._extract_state_dict(video_obj)
            missing, unexpected = model.video_model.load_state_dict(video_state, strict=bool(getattr(args, "revision", None)))
            print(
                f"[INFO] Loaded video branch checkpoint: {video_ckpt} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if unexpected:
                print(f"[WARNING] Video unexpected keys (first 8): {unexpected[:8]}")

    def run(self) -> Dict[str, float]:
        args = self.args
        if args.epochs < 1:
            raise ValueError("epochs must be positive")
        if getattr(args, "revision", None):
            from revision.protocol import fixed_folds
            split = args.revision["split"]
            if split != fixed_folds()[split["fold"] - 1] or args.split_mode != "actor":
                raise ValueError("Invalid revision fold")
            for partition in ("train", "val", "test"):
                if parse_actor_list(getattr(args, f"{partition}_actors")) != split[f"{partition}_actors"]:
                    raise ValueError("Revision actor arguments disagree with fold")
            if not args.use_wavlm or args.num_classes != 8:
                raise ValueError("Revision requires WavLM and eight classes")
        set_seed(args.seed)
        self._init_tracking()

        data_root = Path(args.data_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _is_wsl() and str(data_root.expanduser().resolve()).startswith("/mnt/"):
            print(
                "[WARNING] Data root is on /mnt/* under WSL. "
                "For better training throughput, move data to the Linux filesystem "
                "(e.g. /home/<user>/...)."
            )

        train_actors = parse_actor_list(args.train_actors)
        val_actors = parse_actor_list(args.val_actors)
        test_actors = parse_actor_list(args.test_actors)

        run_dir = Path(getattr(args, "output_dir", "outputs"))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(self._checkpoint_config(), indent=2))
        train_loader, val_loader, test_loader, sizes = build_dataloaders(
            data_root=data_root,
            num_classes=args.num_classes,
            num_frames=args.frames,
            train_actors=train_actors,
            val_actors=val_actors,
            test_actors=test_actors,
            batch_size=args.batch_size,
            device=device,
            requested_num_workers=args.num_workers,
            stratified=(args.split_mode == "stratified"),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            use_wavlm=args.use_wavlm,
            train_augment=not getattr(args, "smoke", False),
            use_face_crop=args.use_face_crop,
            pairs_csv=run_dir / "pairs.csv",
        )

        print(f"\n{'=' * 60}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

        if args.fusion_align_mode != "none" and args.fusion not in {"concat", "gated"}:
            print(
                f"[INFO] fusion_align_mode={args.fusion_align_mode} is only applied to concat/gated fusion; "
                f"current fusion={args.fusion} will ignore it."
            )

        model = build_model(
            args.num_classes,
            args.fusion,
            pretrained_video=not args.no_pretrained_video,
            smoke=getattr(args, "smoke", False),
            xattn_head=getattr(args, "xattn_head", "concat"),
            xattn_d_model=getattr(args, "xattn_d_model", 128),
            xattn_heads=getattr(args, "xattn_heads", 4),
            xattn_attn_dropout=getattr(args, "xattn_attn_dropout", 0.1),
            xattn_stochastic_depth=getattr(args, "xattn_stochastic_depth", 0.1),
            xattn_use_emotion_prior=getattr(args, "xattn_use_emotion_prior", False),
            xattn_emotion_prior_dim=getattr(args, "xattn_emotion_prior_dim", 8),
            xattn_emotion_prior_hidden_dim=getattr(args, "xattn_emotion_prior_hidden_dim", 64),
            xattn_emotion_prior_dropout=getattr(args, "xattn_emotion_prior_dropout", 0.1),
            temporal_pooling=getattr(args, "temporal_pooling", "mean"),
            temporal_num_heads=getattr(args, "temporal_num_heads", 4),
            temporal_num_layers=getattr(args, "temporal_num_layers", 1),
            temporal_dropout=getattr(args, "temporal_dropout", 0.1),
            audio_n_mels=getattr(args, "audio_n_mels", 64),
            use_resnet_audio=getattr(args, "use_resnet_audio", True),
            use_wavlm=args.use_wavlm,
            fusion_align_mode=getattr(args, "fusion_align_mode", "none"),
            fusion_align_dim=getattr(args, "fusion_align_dim", 256),
            fusion_align_temperature=getattr(args, "fusion_align_temperature", 0.07),
        )
        self._load_fusion_branch_checkpoints(model)
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})\n")

        if args.fusion == "late":
            loss_fn = nn.NLLLoss()
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=max(0.0, float(args.label_smoothing)))
        two_stage_enabled = self._is_two_stage_fusion_enabled(model)
        current_stage = 0
        stage1_epochs = 0

        if two_stage_enabled:
            if args.epochs <= 1:
                stage1_epochs = args.epochs
            else:
                stage1_epochs = min(max(1, args.stage1_epochs), args.epochs - 1)
            stage2_epochs = max(0, args.epochs - stage1_epochs)
            current_stage = 1
            self._apply_two_stage_freeze_policy(model, stage=1)
            optimizer = self._build_optimizer(model, stage=1)
            if args.cosine_stage2_only:
                scheduler = None
                if args.use_cosine_annealing:
                    print("[INFO] cosine_stage2_only enabled: stage-1 uses fixed LR.")
            else:
                scheduler = self._build_scheduler(optimizer, epochs_in_stage=stage1_epochs)
            print(
                "[INFO] Two-stage fusion enabled: "
                f"stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}, "
                f"video_unfreeze_blocks={args.fusion_unfreeze_video_blocks}, "
                f"wavlm_unfreeze_layers={args.fusion_unfreeze_wavlm_layers}"
            )
        else:
            optimizer = self._build_optimizer(model, stage=0)
            scheduler = self._build_scheduler(optimizer, epochs_in_stage=args.epochs)

        configured_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Trainable parameters after stage setup: {configured_trainable:,}")

        best_f1 = -1.0
        best_path = run_dir / ("best.pt" if getattr(args, "revision", None) else f"best_{args.fusion}.pt")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        early_stopping_counter = 0

        for epoch in range(1, args.epochs + 1):
            if (
                two_stage_enabled
                and current_stage == 1
                and stage1_epochs < args.epochs
                and epoch == stage1_epochs + 1
            ):
                current_stage = 2
                self._apply_two_stage_freeze_policy(model, stage=2)
                optimizer = self._build_optimizer(model, stage=2)
                scheduler = self._build_scheduler(optimizer, epochs_in_stage=args.epochs - stage1_epochs)
                print(f"[INFO] Switched to stage-2 at epoch {epoch}.")

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_fn,
                args.fusion,
                fusion_align_weight=getattr(args, "fusion_align_weight", 0.0),
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_fn,
                args.fusion,
                fusion_align_weight=getattr(args, "fusion_align_weight", 0.0),
            )

            if scheduler is not None:
                scheduler.step()
            current_lr_text = self._optimizer_lr_text(optimizer)
            stage_text = str(current_stage) if two_stage_enabled else "-"

            print(
                f"Epoch {epoch:02d} | "
                f"stage {stage_text} | "
                f"train loss {train_metrics['loss']:.4f} "
                f"(cls {train_metrics['cls_loss']:.4f}, ctr {train_metrics['contrastive_loss']:.4f}) "
                f"acc {train_metrics['acc']:.4f} f1 {train_metrics['f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} "
                f"(cls {val_metrics['cls_loss']:.4f}, ctr {val_metrics['contrastive_loss']:.4f}) "
                f"acc {val_metrics['acc']:.4f} f1 {val_metrics['f1']:.4f} | "
                f"lr {current_lr_text}"
            )

            if args.wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["loss"],
                        "train/cls_loss": train_metrics["cls_loss"],
                        "train/contrastive_loss": train_metrics["contrastive_loss"],
                        "train/acc": train_metrics["acc"],
                        "train/f1": train_metrics["f1"],
                        "val/loss": val_metrics["loss"],
                        "val/cls_loss": val_metrics["cls_loss"],
                        "val/contrastive_loss": val_metrics["contrastive_loss"],
                        "val/acc": val_metrics["acc"],
                        "val/f1": val_metrics["f1"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "stage": current_stage if two_stage_enabled else 0,
                    }
                )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                early_stopping_counter = 0
                torch.save(
                    {"model": model.state_dict(), "val_f1": best_f1, "epoch": epoch, "config": self._checkpoint_config()},
                    best_path,
                )
            else:
                early_stopping_counter += 1
                if args.early_stopping_patience > 0 and early_stopping_counter >= args.early_stopping_patience:
                    print(f"\nEarly stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
                    print(f"Best val F1: {best_f1:.4f}")
                    break

        best = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        print(f"[INFO] Reloaded best validation checkpoint: {best_path} (epoch {best['epoch']})")
        test_metrics = {}
        if sizes["test"] > 0:
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                loss_fn,
                args.fusion,
                fusion_align_weight=getattr(args, "fusion_align_weight", 0.0),
            )
            print(
                f"Test | loss {test_metrics['loss']:.4f} "
                f"(cls {test_metrics['cls_loss']:.4f}, ctr {test_metrics['contrastive_loss']:.4f}) "
                f"acc {test_metrics['acc']:.4f} f1 {test_metrics['f1']:.4f}"
            )

            record = {"test": test_metrics, "best_epoch": best["epoch"],
                      "val_macro_f1": best["val_f1"], "checkpoint": str(best_path.resolve()),
                      "config": self._checkpoint_config()}
            (run_dir / "metrics.json").write_text(json.dumps(record, indent=2))

            if args.wandb and not getattr(args, "revision", None):
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    for video, audio, labels, _ in test_loader:
                        video = video.to(device)
                        audio = audio.to(device)
                        if args.fusion == "audio":
                            outputs = model(audio)
                        elif args.fusion == "video":
                            outputs = model(video)
                        else:
                            outputs = model(video, audio)
                        preds = outputs.argmax(dim=1)
                        test_preds.append(preds)
                        test_targets.append(labels)

                test_preds = torch.cat(test_preds)
                test_targets = torch.cat(test_targets)
                cm_fig = plot_confusion_matrix(test_preds, test_targets, args.num_classes)
                wandb.log({"test/confusion_matrix": wandb.Image(cm_fig)})
                plt.close(cm_fig)
                wandb.log(
                    {
                        "test/loss": test_metrics["loss"],
                        "test/cls_loss": test_metrics["cls_loss"],
                        "test/contrastive_loss": test_metrics["contrastive_loss"],
                        "test/acc": test_metrics["acc"],
                        "test/f1": test_metrics["f1"],
                    }
                )

        print(f"Best val macro-F1: {best_f1:.4f} | checkpoint: {best_path}")
        if args.wandb:
            wandb.finish()
        return test_metrics


def main() -> None:
    args = build_arg_parser().parse_args()
    EmotionTrainer(args).run()


if __name__ == "__main__":
    main()
