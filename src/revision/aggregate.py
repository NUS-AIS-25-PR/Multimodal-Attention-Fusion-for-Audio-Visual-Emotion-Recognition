"""Aggregate comparable per-fold records, retaining raw values and coverage."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from statistics import mean, stdev
import math

from revision.protocol import (fixed_folds, MODELS, PROTOCOL, run_path,
                               expected_run_config, validate_completed_run)

METRICS = ("accuracy", "precision", "recall", "macro_f1")


def aggregate(root: Path) -> dict:
    result = {"std_policy": "sample (ddof=1); null for a single fold", "methods": {}}
    family = None
    for model in MODELS:
        rows = []
        for split in fixed_folds():
            path = run_path(root, split["fold"], model) / "metrics.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            config = record["config"]
            revision = config["revision"]
            if revision["protocol"] != PROTOCOL or revision["split"] != split or config["fusion"] != model:
                raise ValueError(f"Invalid fold provenance: {path}")
            validate_completed_run(path.parent, expected_run_config(revision, model))
            identity = {k: v for k, v in revision.items() if k != "split"}
            if family is not None and family != identity:
                raise ValueError("Cannot aggregate mixed data/config/smoke experiments")
            family = identity
            row = {"fold": split["fold"], **{k: record["test"][k] for k in METRICS}}
            if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
                raise ValueError(f"Invalid metric: {path}")
            rows.append(row)
        if rows:
            result["methods"][model] = {"folds": rows, "complete_six_folds": len(rows) == 6,
                "summary": {k: {"mean": mean(r[k] for r in rows),
                                 "std": stdev(r[k] for r in rows) if len(rows) > 1 else None}
                            for k in METRICS}}
    result["experiment"] = family
    text = json.dumps(result, indent=2)
    path = root / "aggregate.json"
    if not path.exists() or path.read_bytes() != text.encode():
        path.write_bytes(text.encode())
    with io.StringIO(newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "n_folds", "complete_six_folds"] +
                        [f"{k}_{stat}" for k in METRICS for stat in ("mean", "std")])
        for model, data in result["methods"].items():
            writer.writerow([model, len(data["folds"]), data["complete_six_folds"]] +
                            [data["summary"][k][stat] for k in METRICS for stat in ("mean", "std")])
        text = stream.getvalue()
    path = root / "aggregate.csv"
    if not path.exists() or path.read_bytes() != text.encode():
        path.write_bytes(text.encode())
    return result
