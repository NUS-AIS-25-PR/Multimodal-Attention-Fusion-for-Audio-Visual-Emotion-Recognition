from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.ravdess import DATASET_FACTORY, PAIR_SERVICE, SPLIT_SERVICE
from utils.metrics import classification_metrics
from train import build_model


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def _auto_num_workers(data_root: Path, requested: int) -> int:
    if requested >= 0:
        return requested
    if sys.platform == "win32":
        return 0
    if _is_wsl() and str(data_root.expanduser().resolve()).startswith("/mnt/"):
        return 0
    if _is_wsl():
        return 2
    cpu_count = os.cpu_count() or 4
    return min(8, max(2, cpu_count // 2))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, fusion_mode: str) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    for video, audio, labels, _ in loader:
        video = video.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio if fusion_mode == "audio" else video) if fusion_mode in {"audio", "video"} else model(video, audio)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = classification_metrics(all_preds, all_targets, outputs.shape[-1])
    print(metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=None, choices=[4, 8])
    parser.add_argument(
        "--fusion",
        type=str,
        default=None,
        choices=["audio", "video", "late", "concat", "gated", "xattn", "xattn_concat", "xattn_gated", "chumachenko_ia"],
    )
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_actors", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1 for auto)")
    return parser


class EmotionEvaluator:
    """Evaluation orchestrator for checkpoint-based model validation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> dict:
        args = self.args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = ckpt.get("config", {})
        revision = config.get("revision")
        if revision:
            from revision.protocol import fixed_folds
            from revision.audit import audit_dataset, require_complete
            split = revision["split"]
            if split != fixed_folds()[split["fold"] - 1]:
                raise ValueError("Checkpoint has an invalid revision split")
            report = audit_dataset(Path(args.data_root))
            if report["fingerprint"] != revision["dataset_fingerprint"]:
                raise ValueError("Evaluation dataset differs from checkpoint audit")
            if not revision["smoke"]:
                require_complete(report)
            expected = split["test_actors"]
            if args.test_actors is not None and [int(x) for x in args.test_actors.split(",")] != expected:
                raise ValueError("Test actors disagree with checkpoint fold")
            test_actors = expected
        else:
            test_actors = [int(x) for x in (args.test_actors or "22,23,24").split(",")]
        for key, default in (("fusion", "audio"), ("num_classes", 8), ("frames", 8)):
            supplied = getattr(args, key)
            if supplied is not None and key in config and supplied != config[key]:
                raise ValueError(f"{key} disagrees with checkpoint configuration")
            setattr(args, key, config.get(key, supplied if supplied is not None else default))
        pairs = PAIR_SERVICE.build_pairs(Path(args.data_root))
        _, _, test_pairs = SPLIT_SERVICE.by_actor(pairs, [], [], test_actors)
        if not test_pairs:
            raise ValueError("Empty test partition")
        test_ds = DATASET_FACTORY.create(
            test_pairs, num_classes=args.num_classes, num_frames=args.frames,
            augment=False, use_face_crop=config.get("use_face_crop", True),
            use_wavlm=config.get("use_wavlm", False),
        )
        num_workers = _auto_num_workers(Path(args.data_root), args.num_workers)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, **loader_kwargs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            args.num_classes,
            args.fusion,
            pretrained_video=False,
            checkpoint_init=True,
            smoke=config.get("smoke", False),
            xattn_head=config.get("xattn_head", "concat"),
            xattn_d_model=config.get("xattn_d_model", 128),
            xattn_heads=config.get("xattn_heads", 4),
            xattn_attn_dropout=config.get("xattn_attn_dropout", 0.1),
            xattn_stochastic_depth=config.get("xattn_stochastic_depth", 0.1),
            xattn_use_emotion_prior=config.get("xattn_use_emotion_prior", False),
            xattn_emotion_prior_dim=config.get("xattn_emotion_prior_dim", 8),
            xattn_emotion_prior_hidden_dim=config.get("xattn_emotion_prior_hidden_dim", 64),
            xattn_emotion_prior_dropout=config.get("xattn_emotion_prior_dropout", 0.1),
            temporal_pooling=config.get("temporal_pooling", "mean"),
            temporal_num_heads=config.get("temporal_num_heads", 4),
            temporal_num_layers=config.get("temporal_num_layers", 1),
            temporal_dropout=config.get("temporal_dropout", 0.1),
            audio_n_mels=config.get("audio_n_mels", 64),
            use_resnet_audio=config.get("use_resnet_audio", True),
            use_wavlm=config.get("use_wavlm", False),
            fusion_align_mode=config.get("fusion_align_mode", "none"),
            fusion_align_dim=config.get("fusion_align_dim", 256),
            fusion_align_temperature=config.get("fusion_align_temperature", 0.07),
        )
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.num_classes = args.num_classes
        return evaluate(model, test_loader, device, args.fusion)


def main() -> None:
    args = build_arg_parser().parse_args()
    EmotionEvaluator(args).run()


if __name__ == "__main__":
    main()
