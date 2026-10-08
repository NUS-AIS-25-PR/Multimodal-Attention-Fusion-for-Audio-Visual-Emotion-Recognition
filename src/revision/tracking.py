"""Optional scalar-only W&B transport. Local scientific artifacts are authoritative."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
import uuid

import numpy as np
import torch


@dataclass(frozen=True)
class TrackingOptions:
    mode: str = "disabled"
    project: str = "ieee-spmb-2026"
    group: str | None = None
    entity: str | None = None

    def __post_init__(self):
        if self.mode not in {"disabled", "offline", "online"}:
            raise ValueError("W&B mode must be disabled, offline or online")


@contextmanager
def preserve_rng():
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class RunTracker:
    def __init__(self, path: Path, config: dict, options: TrackingOptions):
        self.path = Path(path) / "tracking.json"
        self.run = None
        revision = config.get("revision", {})
        fold = revision.get("split", {}).get("fold")
        commit = revision.get("git_commit", "unknown")
        group = options.group or (f"spmb2026-{commit[:8]}-{revision.get('dataset_fingerprint', 'local')[:8]}-"
                                  f"{Path(revision.get('output_root', str(path))).name}")
        self.metadata = {"mode": options.mode, "project": options.project, "entity": options.entity,
                         "group": group, "name": f"fold-{fold:02d}-{config['fusion']}" if fold else config["fusion"],
                         "fold": fold, "model": config["fusion"], "profile_id": config.get("profile_id"),
                         "split": revision.get("split"), "git_commit": commit,
                         "config": config, "status": "disabled", "run_id": None}
        self._save()
        if options.mode == "disabled":
            return  # W&B is not even imported in disabled mode.
        try:
            with preserve_rng():
                sdk = importlib.import_module("wandb")
                self.metadata.update(status="initializing", run_id=uuid.uuid4().hex)
                self.run = sdk.init(project=options.project, entity=options.entity, group=group,
                    name=self.metadata["name"], id=self.metadata["run_id"], resume="never", mode=options.mode,
                    dir=str(path), save_code=False, settings=sdk.Settings(disable_git=True, init_timeout=30),
                    config={"fold": fold, "model": config["fusion"], "profile_id": config.get("profile_id"),
                            "split": revision.get("split"), "git_commit": commit, "configuration": config})
                self.run.define_metric("epoch")
                for prefix in ("train/*", "val/*", "lr/*"):
                    self.run.define_metric(prefix, step_metric="epoch")
            self.metadata["status"] = "active"
        except Exception as exc:
            self._unavailable(exc)
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.metadata, indent=2))

    def _unavailable(self, exc):
        # Exception messages can include credential-bearing requests; never print/store them.
        print(f"[WARNING] W&B unavailable ({type(exc).__name__}); local recording continues.", flush=True)
        self.metadata.update(status="unavailable", error_type=type(exc).__name__)
        self.finish()

    def log(self, values: dict):
        if self.run is None:
            return
        try:
            with preserve_rng():
                self.run.log(values)
        except Exception as exc:
            self._unavailable(exc)
            self._save()

    def log_epoch(self, row: dict):
        values = {"epoch": row["epoch"], "stage": row["stage"]}
        for partition in ("train", "val"):
            for metric in ("loss", "accuracy", "precision", "recall", "macro_f1"):
                values[f"{partition}/{metric}"] = row[f"{partition}_{metric}"]
        values.update({f"lr/group_{i}": lr for i, lr in enumerate(json.loads(row["learning_rates"]))})
        self.log(values)

    def finish(self):
        run, self.run = self.run, None
        if run is not None:
            try:
                with preserve_rng():
                    run.finish()
                if self.metadata["status"] == "active":
                    self.metadata["status"] = "finished"
            except Exception as exc:
                self.metadata.update(status="unavailable", error_type=type(exc).__name__)
                print(f"[WARNING] W&B finish unavailable ({type(exc).__name__}); local artifacts retained.")
        self._save()
