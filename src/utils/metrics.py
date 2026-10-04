from __future__ import annotations

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    return (preds == targets).float().mean().item()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def classification_metrics(preds: torch.Tensor, targets: torch.Tensor,
                           num_classes: int = 8) -> dict[str, float]:
    """Macro scores over the fixed class vocabulary; undefined terms are zero."""
    if targets.numel() == 0:
        raise ValueError("Cannot score an empty partition")
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.detach().cpu().numpy(), preds.detach().cpu().numpy(),
        labels=list(range(num_classes)), average="macro", zero_division=0)
    return {"accuracy": accuracy(preds, targets), "precision": float(precision),
            "recall": float(recall), "macro_f1": float(f1)}
