"""Publication figures from validated local artifacts; no W&B or model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from revision.artifacts import (ARTIFACT_SCHEMA, CLASS_NAMES, confusion_counts,
                                validate_local_artifacts)
from revision.protocol import MODELS, expected_run_config, fixed_folds, validate_completed_run

LABELS = {"audio": "Audio", "video": "Video", "concat": "Concat", "gated": "Gated fusion",
          "xattn": "Cross-attention", "chumachenko_ia": "Adapted Intermediate Attention"}
METRICS = {"accuracy": "Accuracy", "precision": "Macro precision", "recall": "Macro recall",
           "macro_f1": "Macro-F1"}
STYLE = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
         "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300}


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals != 0)


def load_records(root: Path, models: list[str], allow_smoke: bool = False) -> dict:
    records = {model: [] for model in models}
    identity = None
    if not (root / "mode.json").is_file():
        raise ValueError(f"Missing experiment mode.json: {root}")
    mode = json.loads((root / "mode.json").read_text())
    for fold in fixed_folds():
        for model in models:
            path = root / f"fold_{fold['fold']:02d}" / model
            if not path.exists():
                continue
            try:
                config = json.loads((path / "config.json").read_text())
                revision = config["revision"]
                if (revision.get("artifact_schema") != ARTIFACT_SCHEMA or revision["split"] != fold
                        or config["fusion"] != model or mode != {"smoke": revision["smoke"], "protocol": revision["protocol"]}):
                    raise ValueError(f"Unsupported/mismatched figure provenance: {path}")
                if revision["smoke"] and not allow_smoke:
                    raise ValueError("Synthetic smoke artifacts cannot produce paper figures. Use --allow-smoke for marked previews only.")
                metrics = validate_completed_run(path, expected_run_config(revision, model))
                history, predictions = validate_local_artifacts(path, config, metrics)
                family = {k: v for k, v in revision.items() if k != "split"}
                if identity is not None and family != identity:
                    raise ValueError("Cannot plot mixed source/config/data/smoke experiments")
                identity = family
                records[model].append({"path": path, "fold": fold["fold"], "metrics": metrics,
                                       "history": history, "predictions": predictions})
            except (OSError, KeyError, TypeError) as exc:
                raise ValueError(f"Missing/incomplete figure input: {path}") from exc
    records = {model: runs for model, runs in records.items() if runs}
    if not records:
        raise ValueError("No completed model/fold artifacts to plot")
    return {"experiment": identity, "records": records}


def comparison_summary(records: dict) -> dict:
    return {model: {"folds": [r["fold"] for r in runs], "n_folds": len(runs),
                    "missing_folds": sorted(set(range(1, 7)) - {r["fold"] for r in runs}),
                    "complete": len(runs) == 6,
                    "metrics": {metric: {"mean": mean(r["metrics"]["test"][metric] for r in runs),
                                         "std": stdev(r["metrics"]["test"][metric] for r in runs) if len(runs) > 1 else None}
                                for metric in METRICS}}
            for model, runs in records.items()}


def pooled_counts(runs: list[dict]) -> np.ndarray:
    seen_samples, seen_actors = set(), set()
    all_predictions = []
    for run in runs:
        rows = run["predictions"]
        actors = {r["actor"] for r in rows}
        keys = {r["sample_id"] for r in rows}
        if seen_samples & keys or seen_actors & actors:
            raise ValueError("Cannot pool repeated samples/actors across held-out folds")
        seen_samples.update(keys)
        seen_actors.update(actors)
        all_predictions.extend(rows)
    return confusion_counts(all_predictions)


def _export(fig, output: Path, name: str) -> list[str]:
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _coverage(n: int) -> str:
    return "6/6 folds" if n == 6 else f"INCOMPLETE: {n}/6 folds"


def _curves(run: dict, title: str):
    rows = run["history"]
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6), layout="constrained")
    for ax, metric in zip(axes.flat, ("loss", *METRICS)):
        for partition, color, style in (("train", "#0072B2", "-"), ("val", "#D55E00", "--")):
            ax.plot(epochs, [r[f"{partition}_{metric}"] for r in rows],
                    label="Train" if partition == "train" else "Validation", color=color, linestyle=style,
                    marker="o", markersize=2.5)
        ax.set_ylabel("Cross-entropy loss" if metric == "loss" else METRICS[metric])
        if metric != "loss":
            ax.set_ylim(0, 1)
        ax.legend()
    ax = axes.flat[-1]
    for group in range(max(len(r["learning_rates"]) for r in rows)):
        ax.plot(epochs, [r["learning_rates"][group] if group < len(r["learning_rates"]) else np.nan for r in rows],
                label=f"Optimizer group {group}", marker="o", markersize=2.5)
    ax.set_ylabel("Learning rate used in epoch")
    ax.set_yscale("log")
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.axvline(run["metrics"]["best_epoch"], color="#009E73", linestyle=":", linewidth=1)
        for previous, current in zip(rows, rows[1:]):
            if previous["stage"] != current["stage"]:
                ax.axvline(current["epoch"] - 0.5, color="0.6", linestyle="--", linewidth=0.8)
    fig.suptitle(title + "\nDotted green: best validation epoch; dashed gray: training stage change")
    return fig


def _confusion(counts: np.ndarray, title: str):
    normalized = normalize_counts(counts)
    fig, ax = plt.subplots(figsize=(7.4, 6.6), layout="constrained")
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(8), CLASS_NAMES)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("Ground-truth emotion")
    for i in range(8):
        for j in range(8):
            text = f"{normalized[i, j]:.2f}" if counts[i].sum() else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, label="Proportion within ground-truth class")
    ax.set_title(title + "\nRow normalized; — denotes a class with no held-out samples")
    return fig


def _comparison(summary: dict, smoke: bool):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    models = list(summary)
    x = np.arange(len(models))
    labels = [f"{LABELS[m]}\n({summary[m]['n_folds']}/6 folds)" for m in models]
    for ax, metric in zip(axes.flat, METRICS):
        values = [summary[m]["metrics"][metric]["mean"] for m in models]
        ax.bar(x, values, color="#0072B2", width=0.65)
        for i, model in enumerate(models):
            std = summary[model]["metrics"][metric]["std"]
            if std is not None:
                ax.errorbar(i, values[i], yerr=std, color="black", capsize=4, fmt="none")
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(METRICS[metric])
        bounds = [(values[i] - (summary[m]["metrics"][metric]["std"] or 0),
                   values[i] + (summary[m]["metrics"][metric]["std"] or 0)) for i, m in enumerate(models)]
        ax.set_ylim(min(0, min(b[0] for b in bounds)) * 1.05,
                    max(1.05, max(b[1] for b in bounds) * 1.05))
        ax.grid(axis="y", alpha=0.2)
    title = "Model performance: mean ± sample standard deviation (ddof=1)"
    if any(not s["complete"] for s in summary.values()):
        title += "\nINCOMPLETE fold coverage; not a six-fold result. SD unavailable for n=1."
    if smoke:
        title = "SMOKE — NOT FOR PAPER\n" + title
    fig.suptitle(title)
    return fig


def generate_figures(root: Path, output: Path, models: list[str] | None = None,
                     allow_smoke: bool = False) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or any(parent.name.startswith("fold_") for parent in (output, *output.parents) if root in parent.parents):
        raise ValueError("Figure output must not overwrite run artifacts")
    loaded = load_records(root, models or list(MODELS), allow_smoke)
    records, experiment = loaded["records"], loaded["experiment"]
    summary = comparison_summary(records)
    # Validate all pooling before exporting anything.
    pooled = {model: pooled_counts(runs) for model, runs in records.items()}
    output.mkdir(parents=True, exist_ok=True)
    prefix = "smoke_" if experiment["smoke"] else ""
    warning = "SMOKE — NOT FOR PAPER | " if experiment["smoke"] else ""
    exports = []
    inputs = {}
    with plt.rc_context(STYLE):
        for model, runs in records.items():
            coverage = _coverage(len(runs))
            for run in runs:
                name = f"{prefix}fold_{run['fold']:02d}_{model}"
                title = f"{warning}{LABELS[model]} — Fold {run['fold']} | {coverage}"
                exports.extend(_export(_curves(run, title), output, name + "_learning_curves"))
                exports.extend(_export(_confusion(confusion_counts(run["predictions"]), title), output, name + "_confusion"))
                for file in ("metrics.json", "config.json", "history.csv", "test_predictions.csv", "confusion_matrix.json", "tracking.json"):
                    path = run["path"] / file
                    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            title = f"{warning}{LABELS[model]} — pooled held-out predictions | {coverage}"
            exports.extend(_export(_confusion(pooled[model], title), output, f"{prefix}{model}_pooled_confusion"))
        exports.extend(_export(_comparison(summary, experiment["smoke"]), output, prefix + "model_comparison"))
    manifest = {"input_root": str(root), "experiment": experiment, "smoke_not_for_paper": experiment["smoke"],
                "missing_models": [m for m in (models or list(MODELS)) if m not in records],
                "std_policy": "sample (ddof=1); unavailable for n=1", "models": summary,
                "pooled_counts": {m: c.tolist() for m, c in pooled.items()},
                "input_sha256": inputs, "figures": exports}
    (output / f"{prefix}figures_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--allow-smoke", action="store_true", help="Explicitly watermarked previews; never paper figures")
    args = parser.parse_args()
    manifest = generate_figures(args.input_root, args.output_dir, args.model, args.allow_smoke)
    print(f"Exported {len(manifest['figures'])} PDF/SVG/300 DPI PNG files to {args.output_dir}")
    for model, data in manifest["models"].items():
        print(f"{model}: {_coverage(data['n_folds'])}; held-out folds {data['folds']}")
    if manifest["missing_models"]:
        print("No completed folds (0/6), excluded without invented values: " + ", ".join(manifest["missing_models"]))


if __name__ == "__main__":
    main()
