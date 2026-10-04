"""Fixed actor protocol and auditable checkpoint provenance."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import torch

PROTOCOL = "ravdess-fixed-six-fold-v1"
MODELS = ("audio", "video", "concat", "chumachenko_ia", "gated", "xattn")


def validate_split(split: dict) -> None:
    groups = [split[f"{name}_actors"] for name in ("train", "val", "test")]
    if [len(g) for g in groups] != [16, 4, 4]:
        raise ValueError("Expected 16/4/4 actors")
    if any(len(set(g)) != len(g) for g in groups):
        raise ValueError("Duplicate actor in partition")
    train, val, test = map(set, groups)
    if train & val or train & test or val & test:
        raise ValueError("Actor leakage between partitions")
    if train | val | test != set(range(1, 25)):
        raise ValueError("Actor union must be 1..24")


def fixed_folds() -> list[dict]:
    groups = [list(range(i, i + 4)) for i in range(1, 25, 4)]
    folds = []
    for i in range(6):
        test, val = groups[i], groups[(i + 1) % 6]
        split = {"fold": i + 1, "train_actors": sorted(set(range(1, 25)) - set(test + val)),
                 "val_actors": val, "test_actors": test}
        validate_split(split)
        folds.append(split)
    for name in ("val", "test"):
        if Counter(a for f in folds for a in f[f"{name}_actors"]) != Counter(range(1, 25)):
            raise ValueError(f"Invalid {name} actor coverage")
    return folds


def run_path(root: Path, fold: int, model: str) -> Path:
    if fold not in range(1, 7) or model not in MODELS:
        raise ValueError("Unknown fold/model")
    return Path(root) / f"fold_{fold:02d}" / model


def validate_warm_start(path: Path, revision: dict, modality: str) -> None:
    expected = run_path(Path(revision["output_root"]), revision["split"]["fold"], modality) / "best.pt"
    if path.resolve() != expected.resolve():
        raise ValueError(f"Forbidden old/cross-fold checkpoint: {path}; expected {expected}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    config = obj.get("config", {})
    source = config.get("revision")
    if source != revision or config.get("fusion") != modality or not config.get("use_wavlm"):
        raise ValueError(f"Checkpoint provenance mismatch: {path}")
    validate_split(source["split"])
