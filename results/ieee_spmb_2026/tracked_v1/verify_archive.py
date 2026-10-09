"""Verify the published result snapshot without training, inference, or dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METHODS = ("audio", "video", "gated", "chumachenko_ia", "xattn")
SCORES = ("accuracy", "precision", "recall", "macro_f1")
CLASSES = ("Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(archive: Path, check_local_checkpoints: bool = False) -> dict:
    inventory = read_json(archive / "artifact_inventory.json")
    declared = {}
    for record in inventory["files"]:
        relative = record["archive"]
        path = archive / relative
        assert path.resolve().is_relative_to(archive.resolve()), relative
        assert relative not in declared, f"Duplicate archive path: {relative}"
        assert path.is_file() and not path.is_symlink(), relative
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {relative}"
        assert sha256(path) == record["sha256"], f"SHA256 mismatch: {relative}"
        assert path.suffix not in {".pt", ".pth", ".ckpt", ".mp4", ".wav"}, relative
        declared[relative] = record
    checkpoints = inventory["local_only_checkpoints"]
    assert len(checkpoints) == 31
    assert sum(r["run_status"] == "completed_best_validation" for r in checkpoints) == 30
    assert inventory["checkpoint_disposition"] == "retained_local_by_user_request"
    if check_local_checkpoints:
        for record in checkpoints:
            path = Path(record["source"])
            assert path.stat().st_size == record["bytes"], path
            assert sha256(path) == record["sha256"], path

    root = archive / "outputs/speaker_independent_tracked_v1"
    aggregate = read_json(root / "aggregate.json")
    report = read_json(root / "six_fold_report.json")
    assert report["complete"] and not report["failed_or_partial_folds"]
    assert set(aggregate["methods"]) == set(METHODS)
    assert read_json(root / "mode.json") == {
        "smoke": False, "protocol": "ravdess-fixed-six-fold-v1"}
    audit = read_json(next(root.glob("audit-*.json")))
    assert audit["complete"] and audit["total_pairs"] == 1440
    assert all(audit["by_actor"][str(a)]["pairs"] == 60 for a in range(1, 25))
    expected_keys = {
        (1, emotion, intensity, statement, repetition, actor)
        for actor in range(1, 25) for emotion in range(1, 9)
        for intensity in ([1] if emotion == 1 else [1, 2])
        for statement in (1, 2) for repetition in (1, 2)}
    per_fold = read_csv(root / "per_fold_metrics.csv")
    assert len(per_fold) == 30
    assert len({(row["method"], row["fold"]) for row in per_fold}) == 30
    pooled = {model: [[0] * 8 for _ in range(8)] for model in METHODS}
    observed = {model: [] for model in METHODS}
    seen = {model: set() for model in METHODS}
    family = aggregate["experiment"]
    assert family["git_commit"] == inventory["training_git_commit"]
    assert family["dataset_fingerprint"] == audit["fingerprint"]
    assert family["git_commit"] == report["git_commit"]
    for fold in range(1, 7):
        group = list(range(4 * fold - 3, 4 * fold + 1))
        next_fold = fold % 6 + 1
        validation = list(range(4 * next_fold - 3, 4 * next_fold + 1))
        split = {"fold": fold, "test_actors": group, "val_actors": validation,
                 "train_actors": sorted(set(range(1, 25)) - set(group + validation))}
        revision = read_json(root / f"fold_{fold:02d}/split.json")
        assert revision == {**family, "split": split}
        for model in METHODS:
            path = root / f"fold_{fold:02d}" / model
            config = read_json(path / "config.json")
            metrics = read_json(path / "metrics.json")
            tracking = read_json(path / "tracking.json")
            assert config == metrics["config"] == tracking["config"]
            assert config["revision"] == revision and config["fusion"] == model
            assert config["profile_id"] == family["profiles"][model]["id"]
            assert config["resolved_profile"] == family["profiles"][model]["config"]
            assert config["output_dir"] == str(Path(family["output_root"]) / f"fold_{fold:02d}" / model)
            assert (tracking["mode"], tracking["status"], tracking["project"], tracking["group"]) == (
                "offline", "finished", "ieee-spmb-2026", "spmb2026-canonical-tracked-v1")
            for modality in ("audio", "video"):
                expected = "" if model in {"audio", "video"} else str(
                    Path(family["output_root"]) / f"fold_{fold:02d}" / modality / "best.pt")
                assert config[f"{modality}_ckpt"] == expected
            checkpoint = next(r for r in checkpoints if r["source"] == metrics["checkpoint"])
            assert checkpoint["run_status"] == "completed_best_validation"
            history = read_csv(path / "history.csv")
            assert [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
            assert 1 <= len(history) <= config["epochs"]
            for row in history:
                assert int(row["stage"]) in (0, 1, 2)
                assert all(math.isfinite(float(row[f"{part}_{score}"]))
                           for part in ("train", "val") for score in ("loss", *SCORES))
                rates = json.loads(row["learning_rates"])
                assert rates and all(math.isfinite(lr) and lr >= 0 for lr in rates)
            best = max(history, key=lambda row: float(row["val_macro_f1"]))
            assert int(best["epoch"]) == metrics["best_epoch"]
            assert float(best["val_macro_f1"]) == metrics["val_macro_f1"]
            predictions = read_csv(path / "test_predictions.csv")
            assert len(predictions) == 240
            keys = set()
            counts = [[0] * 8 for _ in range(8)]
            for row in predictions:
                key = tuple(map(int, row["sample_id"].split("-")))
                actor, label, prediction = (int(row[k]) for k in ("actor", "label", "prediction"))
                assert key in expected_keys and key[-1] == actor and actor in group
                assert label == key[1] - 1 and 0 <= prediction < 8
                assert key not in keys and key not in seen[model]
                keys.add(key)
                counts[label][prediction] += 1
            assert keys == {k for k in expected_keys if k[-1] in group}
            seen[model].update(keys)
            confusion = read_json(path / "confusion_matrix.json")
            assert confusion["counts"] == counts, f"Confusion matrix disagrees with predictions: {model}/{fold}"
            assert confusion["class_names"] == list(CLASSES)
            precision, recall, f1 = [], [], []
            for i in range(8):
                tp, actual, predicted = counts[i][i], sum(counts[i]), sum(row[i] for row in counts)
                precision.append(tp / predicted if predicted else 0.0)
                recall.append(tp / actual if actual else 0.0)
                f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
                for j in range(8):
                    pooled[model][i][j] += counts[i][j]
            scores = dict(zip(SCORES, (sum(counts[i][i] for i in range(8)) / 240,
                                      mean(precision), mean(recall), mean(f1))))
            assert all(math.isclose(scores[k], metrics["test"][k], abs_tol=1e-7) for k in SCORES)
            csv_row = next(r for r in per_fold if r["method"] == model and int(r["fold"]) == fold)
            assert all(float(csv_row[k]) == metrics["test"][k] for k in SCORES)
            assert int(csv_row["best_epoch"]) == metrics["best_epoch"]
            observed[model].append(metrics["test"])
    figures = read_json(archive / "outputs/paper_figures_six_fold_tracked_v1/figures_manifest.json")
    assert set(figures["models"]) == set(METHODS) and not figures["missing_models"]
    assert not figures["smoke_not_for_paper"] and len(figures["figures"]) == 198
    for model in METHODS:
        assert seen[model] == expected_keys
        assert figures["pooled_counts"][model] == pooled[model]
        assert figures["models"][model]["complete"] and figures["models"][model]["n_folds"] == 6
        assert aggregate["methods"][model]["complete_six_folds"]
        for score in SCORES:
            summary = aggregate["methods"][model]["summary"][score]
            values = [r[score] for r in observed[model]]
            assert math.isclose(summary["mean"], mean(values), abs_tol=1e-12)
            assert math.isclose(summary["std"], stdev(values), abs_tol=1e-12)
    for original_path in figures["figures"]:
        path = archive / "outputs/paper_figures_six_fold_tracked_v1" / Path(original_path).name
        assert path.is_file() and path.stat().st_size > 1000
        if path.suffix == ".svg":
            assert "INCOMPLETE" not in path.read_text() and "NOT FOR PAPER" not in path.read_text()
    return {"status": "PASS", "copied_artifacts": len(declared), "completed_runs": 30,
            "held_out_samples_per_method": 1440, "figure_exports": 198,
            "local_checkpoint_hashes_checked": len(checkpoints) if check_local_checkpoints else 0,
            "note": "Archive-only verification recomputes metrics; checkpoint binaries require the optional local check."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-local-checkpoints", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.check_local_checkpoints), indent=2))
