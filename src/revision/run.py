"""Actor-independent revision runner. No legacy checkpoint inputs are accepted."""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from revision.audit import audit_dataset, require_complete
from revision.aggregate import aggregate
from revision.protocol import fixed_folds, MODELS, PROTOCOL, run_path
from revision.smoke import create_smoke_media
from train import EmotionTrainer, build_arg_parser as training_parser


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.frames < 1:
        raise ValueError("epochs, batch-size and frames must be positive")
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Mode sentinel makes smoke/production mixing impossible even for disjoint folds.
    mode = root / "mode.json"
    expected_mode = {"smoke": args.smoke, "protocol": PROTOCOL}
    if mode.exists() and json.loads(mode.read_text()) != expected_mode:
        raise ValueError("Output root belongs to a different smoke/production mode")
    mode.write_text(json.dumps(expected_mode))
    if args.smoke:
        torch.set_num_threads(1)
        data_root = root / "synthetic_media"
        if not data_root.exists():
            create_smoke_media(data_root)
    else:
        if args.data_root is None:
            raise ValueError("Production requires --data-root")
        data_root = args.data_root.expanduser().resolve()
    audit = audit_dataset(data_root)
    audit_path = root / f"audit-{audit['fingerprint']}.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    if not args.smoke:
        require_complete(audit)
    elif audit["duplicates"] or audit["invalid_files"] or any(audit["missing_counterparts"].values()):
        raise ValueError("Invalid synthetic fixture")
    folds = fixed_folds()
    selected = folds if args.fold is None else [folds[args.fold - 1]]
    requested = list(MODELS) if args.model == "all" else [args.model]
    # Fusion has automatic same-fold audio/video prerequisites.
    models = [m for m in MODELS if m in requested or
              (any(m0 not in {"audio", "video"} for m0 in requested) and m in {"audio", "video"})]
    settings = {"seed": args.seed, "epochs": 1 if args.smoke else args.epochs,
                "batch_size": args.batch_size, "frames": 2 if args.smoke else args.frames,
                "lr": args.lr, "use_face_crop": not args.no_face_crop and not args.smoke,
                "num_workers": 0 if args.smoke else args.num_workers}
    source_root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        source_hash.update(str(source.relative_to(source_root)).encode())
        source_hash.update(source.read_bytes())
    revision_common = {"source_fingerprint": source_hash.hexdigest(),
                       "protocol": PROTOCOL, "smoke": args.smoke,
                       "output_root": str(root), "dataset_fingerprint": audit["fingerprint"],
                       "data_root": str(data_root), "settings": settings,
                       "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "versions": {k: importlib.metadata.version(k)
                                    for k in ("torch", "torchvision", "transformers", "numpy", "scikit-learn")}}
    # Protect completed work and reject mixed configurations before starting any run.
    for path in root.glob("fold_*/split.json"):
        saved = json.loads(path.read_text())
        if {k: v for k, v in saved.items() if k != "split"} != revision_common:
            raise ValueError(f"Output root contains a different experiment: {path}")
    planned = []
    for split in selected:
        revision = {**revision_common, "split": split}
        for model in models:
            path = run_path(root, split["fold"], model)
            if path.exists():
                # Existing unimodal dependencies may be reused only after provenance validation.
                if model not in requested and (path / "best.pt").exists() and (path / "metrics.json").exists():
                    from revision.protocol import validate_warm_start
                    validate_warm_start(path / "best.pt", revision, model)
                    continue
                raise FileExistsError(f"Run already exists; choose a fresh output root: {path}")
            planned.append((split, model, path, revision))
    for split, model, path, revision in planned:
        path.mkdir(parents=True, exist_ok=False)
        (path.parent / "split.json").write_text(json.dumps(revision, indent=2))
        train_args = training_parser().parse_args(["--data_root", str(data_root)])
        for key, value in settings.items():
            setattr(train_args, key, value)
        train_args.fusion = model
        train_args.split_mode = "actor"
        train_args.use_wavlm = True
        train_args.no_pretrained_video = args.smoke
        train_args.smoke = args.smoke
        train_args.revision = revision
        train_args.output_dir = str(path)
        for partition in ("train", "val", "test"):
            setattr(train_args, f"{partition}_actors", ",".join(map(str, split[f"{partition}_actors"])))
        if model not in {"audio", "video"}:
            train_args.audio_ckpt = str(run_path(root, split["fold"], "audio") / "best.pt")
            train_args.video_ckpt = str(run_path(root, split["fold"], "video") / "best.pt")
        print(f"[REVISION] fold={split['fold']} model={model} smoke={args.smoke}", flush=True)
        print(f"[REVISION] disjoint split={split}", flush=True)
        print(f"[REVISION] audio_ckpt={train_args.audio_ckpt or 'none'} "
              f"video_ckpt={train_args.video_ckpt or 'none'}", flush=True)
        EmotionTrainer(train_args).run()
    return aggregate(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/speaker_independent"))
    parser.add_argument("--fold", type=int, choices=range(1, 7))
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output_root == Path("outputs/speaker_independent"):
        args.output_root = Path("outputs/revision_smoke")
    run(args)


if __name__ == "__main__":
    main()
