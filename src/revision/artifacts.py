"""Durable local epoch and held-out prediction artifacts, independent of W&B."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from utils.metrics import classification_metrics

ARTIFACT_SCHEMA = "spmb-local-tracking-v1"
SCORES = ("loss", "accuracy", "precision", "recall", "macro_f1")
HISTORY_FIELDS = ("epoch", "stage", *(f"{p}_{m}" for p in ("train", "val") for m in SCORES),
                  "learning_rates")
PREDICTION_FIELDS = ("sample_id", "actor", "label", "prediction")
CLASS_NAMES = ("Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised")


class HistoryWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        # Existing histories must never be truncated or appended on a completed-run skip.
        with self.path.open("x", newline="") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, epoch: int, stage: int, train: dict, val: dict, learning_rates: list) -> dict:
        row = {"epoch": epoch, "stage": stage, "learning_rates": json.dumps(learning_rates)}
        for partition, scores in (("train", train), ("val", val)):
            row.update({f"{partition}_{key}": scores[key] for key in SCORES})
        with self.path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=HISTORY_FIELDS).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        return row


def read_history(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(HISTORY_FIELDS):
            raise ValueError(f"Invalid history columns: {path}")
        rows = []
        for raw in reader:
            row = {"epoch": int(raw["epoch"]), "stage": int(raw["stage"]),
                   "learning_rates": json.loads(raw["learning_rates"])}
            for partition in ("train", "val"):
                for score in SCORES:
                    value = float(raw[f"{partition}_{score}"])
                    if not math.isfinite(value) or (score != "loss" and not 0 <= value <= 1):
                        raise ValueError(f"Non-finite/out-of-range history metric: {path}")
                    row[f"{partition}_{score}"] = value
            if (row["stage"] not in (0, 1, 2) or not row["learning_rates"]
                    or any(not math.isfinite(x) or x < 0 for x in row["learning_rates"])):
                raise ValueError(f"Invalid stage/learning rates: {path}")
            rows.append(row)
    if not rows or [r["epoch"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Empty/duplicate/non-contiguous history: {path}")
    return rows


def confusion_counts(rows: list[dict]) -> np.ndarray:
    counts = np.zeros((8, 8), dtype=np.int64)
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        if not (0 <= label < 8 and 0 <= prediction < 8):
            raise ValueError("Predictions must use the eight-class vocabulary")
        counts[label, prediction] += 1
    return counts


def save_test_artifacts(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot save empty held-out predictions")
    counts = confusion_counts(rows)
    with (path / "test_predictions.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    with (path / "confusion_matrix.json").open("x") as stream:
        json.dump({"class_names": CLASS_NAMES, "labels": list(range(8)),
                   "rows": "ground truth", "columns": "prediction", "counts": counts.tolist()}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def read_predictions(path: Path) -> list[dict]:
    with (path / "test_predictions.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(PREDICTION_FIELDS):
            raise ValueError(f"Invalid prediction columns: {path}")
        return [{"sample_id": r["sample_id"], **{k: int(r[k]) for k in PREDICTION_FIELDS[1:]}}
                for r in reader]


def validate_local_artifacts(path: Path, config: dict, metrics: dict) -> tuple[list, list]:
    """Check completion, best-epoch history and unique, strictly held-out speech keys."""
    from revision.audit import expected_keys
    tracking = json.loads((path / "tracking.json").read_text())
    revision = config["revision"]
    if (tracking.get("config") != config or tracking.get("mode") not in {"disabled", "offline", "online"}
            or tracking.get("fold") != revision["split"]["fold"] or tracking.get("model") != config["fusion"]
            or tracking.get("profile_id") != config["profile_id"] or tracking.get("split") != revision["split"]
            or tracking.get("git_commit") != revision.get("git_commit", "unknown")):
        raise ValueError(f"Tracking metadata disagrees with scientific configuration: {path}")
    history = read_history(path / "history.csv")
    if len(history) > config["epochs"]:
        raise ValueError(f"Too many history epochs: {path}")
    best = max(history, key=lambda row: row["val_macro_f1"])
    if best["epoch"] != metrics["best_epoch"] or best["val_macro_f1"] != metrics["val_macro_f1"]:
        raise ValueError(f"History disagrees with selected checkpoint: {path}")
    rows = read_predictions(path)
    test_actors = set(config["revision"]["split"]["test_actors"])
    expected = expected_keys()
    keys = set()
    for row in rows:
        key = tuple(int(x) for x in row["sample_id"].split("-"))
        if (key not in expected or key[-1] != row["actor"] or row["actor"] not in test_actors
                or row["label"] != key[1] - 1 or key in keys):
            raise ValueError(f"Invalid/duplicate/non-held-out prediction: {path}")
        keys.add(key)
    if config["revision"]["smoke"]:
        if len(rows) != 4 or {r["actor"] for r in rows} != test_actors:
            raise ValueError(f"Incomplete smoke predictions: {path}")
    elif keys != {k for k in expected if k[-1] in test_actors}:
        raise ValueError(f"Incomplete production held-out predictions: {path}")
    counts = confusion_counts(rows)
    saved = json.loads((path / "confusion_matrix.json").read_text())
    if (saved.get("counts") != counts.tolist() or saved.get("labels") != list(range(8))
            or saved.get("class_names") != list(CLASS_NAMES)
            or saved.get("rows") != "ground truth" or saved.get("columns") != "prediction"):
        raise ValueError(f"Confusion matrix disagrees with predictions: {path}")
    scores = classification_metrics(torch.tensor([r["prediction"] for r in rows]),
                                    torch.tensor([r["label"] for r in rows]))
    if any(not math.isclose(value, metrics["test"][key], abs_tol=1e-7) for key, value in scores.items()):
        raise ValueError(f"Test metrics disagree with predictions: {path}")
    return history, rows
