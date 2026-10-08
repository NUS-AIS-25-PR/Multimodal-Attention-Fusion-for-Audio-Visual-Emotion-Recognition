"""Fixed actor protocol and auditable checkpoint provenance."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
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


def expected_run_config(revision: dict, model: str) -> dict:
    """Reconstruct the entire expected trainer config, including canonical profile."""
    from revision.profiles import resolve_profiles
    if revision["profiles"] != resolve_profiles(revision["smoke"], revision["settings"]):
        raise ValueError("Revision profile catalog differs from canonical profiles")
    split = revision["split"]
    if split != fixed_folds()[split["fold"] - 1]:
        raise ValueError("Revision split differs from fixed fold")
    profile = revision["profiles"][model]
    root = Path(revision["output_root"])
    config = {**profile["config"], "profile_id": profile["id"],
              "resolved_profile": profile["config"], "revision": revision,
              "data_root": revision["data_root"],
              "output_dir": str(run_path(root, split["fold"], model)),
              "audio_ckpt": "", "video_ckpt": ""}
    for partition in ("train", "val", "test"):
        config[f"{partition}_actors"] = ",".join(map(str, split[f"{partition}_actors"]))
    if model not in {"audio", "video"}:
        for modality in ("audio", "video"):
            config[f"{modality}_ckpt"] = str(run_path(root, split["fold"], modality) / "best.pt")
    return config


def validate_completed_run(path: Path, expected_config: dict) -> dict:
    """Accept only complete, readable artifacts with identical config/provenance."""
    required = [path / name for name in ("best.pt", "metrics.json", "config.json")]
    required.append(path.parent / "split.json")
    from revision.artifacts import ARTIFACT_SCHEMA, validate_local_artifacts
    if expected_config["revision"].get("artifact_schema") == ARTIFACT_SCHEMA:
        required.extend(path / name for name in ("history.csv", "test_predictions.csv", "confusion_matrix.json", "tracking.json"))
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"Incomplete existing run: {path}; missing {', '.join(missing)}")
    try:
        config = json.loads((path / "config.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        split = json.loads((path.parent / "split.json").read_text())
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Unreadable existing run: {path}") from exc
    if (config != expected_config or metrics.get("config") != expected_config
            or best.get("config") != expected_config or split != expected_config["revision"]):
        raise ValueError(f"Existing run provenance/config mismatch: {path}")
    epoch = best.get("epoch")
    val_f1 = best.get("val_f1")
    state = best.get("model")
    if (not isinstance(epoch, int) or not 1 <= epoch <= config["epochs"]
            or not isinstance(val_f1, (int, float)) or not math.isfinite(val_f1) or not 0 <= val_f1 <= 1
            or not isinstance(state, dict) or not state
            or not all(isinstance(value, torch.Tensor) for value in state.values())
            or metrics.get("best_epoch") != epoch or metrics.get("val_macro_f1") != val_f1
            or metrics.get("checkpoint") != str((path / "best.pt").resolve())):
        raise ValueError(f"Invalid existing best-checkpoint evidence: {path}")
    for name in ("accuracy", "precision", "recall", "macro_f1"):
        value = metrics.get("test", {}).get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid existing test metrics: {path}")
    if expected_config["revision"].get("artifact_schema") == ARTIFACT_SCHEMA:
        try:
            validate_local_artifacts(path, config, metrics)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Invalid/incomplete local tracking artifacts: {path}") from exc
    return metrics


def validate_warm_start(path: Path, revision: dict, modality: str) -> None:
    expected = run_path(Path(revision["output_root"]), revision["split"]["fold"], modality) / "best.pt"
    if path.resolve() != expected.resolve():
        raise ValueError(f"Forbidden old/cross-fold checkpoint: {path}; expected {expected}")
    validate_completed_run(path.parent, expected_run_config(revision, modality))
