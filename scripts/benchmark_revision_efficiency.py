"""Checkpoint-only five-model efficiency measurement; never reads RAVDESS media."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METHODS = ("audio", "video", "gated", "xattn", "chumachenko_ia")
WAVEFORM_SHAPE = (1, 48000)
VIDEO_SHAPE = (1, 8, 3, 112, 112)
OPERATION_DEFINITION = (
    "torchinfo 1.8.0 total_mult_adds / 1e9, estimated GMacs, NOT GFLOPs. "
    "Leaf-module parameter/output heuristics include convolution and bias terms. "
    "Linear layers with sequence outputs are counted using batch size only, "
    "under-counting token-wise projections. Functional transformer QK/AV matmuls, "
    "softmax, elementwise modulation, reductions and positional-bias operations "
    "are not reliably counted; non-leaf MultiheadAttention can be omitted. "
    "The same estimator is used for all five models; no MAC-to-FLOP conversion."
)
CAPTION = (
    "Total parameters include all registered encoder and classifier parameters, "
    "including frozen and unused registered heads. Estimated GMacs are torchinfo "
    "1.8.0 total_mult_adds divided by 10^9, not rigorous GFLOPs; sequence Linear "
    "and dynamic attention operations are under-counted. Latency is the measured "
    "mean CUDA-event elapsed time per batch-1 forward on an NVIDIA GeForce RTX "
    "5080 Laptop GPU, in eval() and torch.inference_mode(), FP32 with TF32 and "
    "autocast disabled. Each model uses its strictly restored Fold 1 best.pt, "
    "with a preallocated random waveform [1,48000] and/or video [1,8,3,112,112]. "
    "Three repeats each follow 20 warmups and contain 100 timed forwards; CUDA "
    "is synchronized before and after each timed loop; unadjusted Linux "
    "CLOCK_MONOTONIC_RAW wall time is retained as a cross-check. Disk I/O, model loading, "
    "data loading, face cropping and audio resampling are excluded. No held-out "
    "data or recognition metrics enter this efficiency measurement."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root / "src")).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def packages() -> dict:
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}


def require_environment(snapshot: dict) -> None:
    current = packages()
    if current != snapshot or current.get("mediapipe") != "0.10.21":
        differences = {name: [snapshot.get(name), current.get(name)]
                       for name in snapshot.keys() | current.keys()
                       if snapshot.get(name) != current.get(name)}
        raise RuntimeError(f"STOP: package snapshot mismatch: {differences}")


def artifact_snapshot(root: Path) -> dict:
    """Hash all 30 production directories, including binaries, without writing."""
    result = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_symlink():
            result[key] = {"symlink": str(path.readlink()), "mtime_ns": path.lstat().st_mtime_ns}
        elif path.is_file():
            result[key] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns}
    return result


def model_kwargs(config: dict, builder) -> dict:
    """All architecture arguments come from the saved config, without defaults."""
    kwargs = {}
    for name in inspect.signature(builder).parameters:
        if name == "pretrained_video":
            kwargs[name] = False  # bootstrap only; full state is strictly loaded below
        elif name == "checkpoint_init":
            kwargs[name] = True  # no download, no inference until strict load succeeds
        else:
            kwargs[name] = config[name]
    if kwargs.get("smoke") or kwargs["num_classes"] != 8 or not kwargs["use_wavlm"]:
        raise ValueError("Expected a production eight-class WavLM checkpoint")
    return kwargs


def restore_model(config: dict, checkpoint: dict, builder):
    if checkpoint.get("config") != config or not checkpoint.get("model"):
        raise ValueError("Checkpoint config/state mismatch; random initialization forbidden")
    model = builder(**model_kwargs(config, builder))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def select_inputs(method: str, waveform, video) -> tuple:
    if method == "audio":
        return (waveform,)
    if method == "video":
        return (video,)
    if method in METHODS:
        return (video, waveform)
    raise ValueError(f"Unsupported method: {method}")


def total_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def latency_statistics(values: list[float]) -> dict:
    return {"n": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "sample_stdev_ms": statistics.stdev(values) if len(values) > 1 else None,
            "min_ms": min(values), "max_ms": max(values)}


def measure_latency(model, inputs: tuple, warmup: int, passes: int, repeats: int) -> dict:
    if warmup < 20 or passes < 100 or repeats < 3:
        raise ValueError("Require >=20 warmups, >=100 forwards and >=3 repeats")
    model.eval()
    records = []
    with torch.inference_mode():
        for repeat in range(repeats):
            # Create and initialize event resources outside the timed loop.
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(passes)]
            for start, end in events:
                start.record()
                end.record()
            for _ in range(warmup):
                model(*inputs)
            torch.cuda.synchronize()
            wall_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            adjusted_start = time.perf_counter()
            for start, end in events:
                start.record()
                model(*inputs)
                end.record()
            torch.cuda.synchronize()
            adjusted_ms = (time.perf_counter() - adjusted_start) * 1000
            wall_ms = (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - wall_start) / 1e6
            values = [start.elapsed_time(end) for start, end in events]
            records.append({"repeat": repeat + 1, "warmups": warmup, "forwards": passes,
                            "cuda_event_ms": values, "cuda_event_statistics": latency_statistics(values),
                            "synchronized_wall_loop_ms": wall_ms, "wall_ms_per_sample": wall_ms / passes,
                            "adjusted_perf_counter_loop_ms_diagnostic_only": adjusted_ms,
                            "gpu_status_after_repeat": subprocess.check_output(
                                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu", "--format=csv,noheader"], text=True).strip()})
            print(f"  repeat {repeat + 1}: GPU {statistics.mean(values):.3f} ms, "
                  f"raw wall {wall_ms / passes:.3f} ms/sample", flush=True)
    return {"primary_convention": "CUDA-event elapsed ms per forward; combined mean of all samples",
            "repeats": records,
            "combined": latency_statistics([v for record in records for v in record["cuda_event_ms"]]),
            "wall_repeat_average_statistics": latency_statistics([r["wall_ms_per_sample"] for r in records]),
            "repeat_mean_statistics": latency_statistics([r["cuda_event_statistics"]["mean_ms"] for r in records])}


def operation_profile(model, inputs: tuple, output: Path, method: str) -> dict:
    import torchinfo
    try:
        with torch.inference_mode():
            summary = torchinfo.summary(model, input_data=inputs, verbose=0, mode="eval",
                                        cache_forward_pass=False, device="cuda")
        (output / f"{method}_torchinfo.txt").write_text(str(summary) + "\n")
        coverage = [{"class": layer.class_name, "name": layer.var_name,
                     "output_shape": layer.output_size, "leaf": layer.is_leaf_layer,
                     "executed": layer.executed, "mult_adds": layer.macs}
                    for layer in summary.summary_list]
        return {"status": "estimated", "tool": "torchinfo", "version": torchinfo.__version__,
                "total_mult_adds": summary.total_mult_adds,
                "estimated_gmacs": summary.total_mult_adds / 1e9,
                "definition": OPERATION_DEFINITION, "layer_coverage": coverage}
    except Exception as exc:
        error = traceback.format_exc()
        (output / f"{method}_profiler_error.txt").write_text(error)
        classes = []
        cause = exc
        while cause is not None:
            classes.append(f"{type(cause).__module__}.{type(cause).__name__}")
            cause = cause.__cause__
        print(f"  profiling N/A: {classes}", flush=True)
        return {"status": "N/A", "tool": "torchinfo", "version": torchinfo.__version__,
                "total_mult_adds": None, "estimated_gmacs": None,
                "error_classes": classes, "error": error, "definition": OPERATION_DEFINITION}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    training_root, output = args.training_root.resolve(), args.output_root.resolve()
    production = training_root / "outputs/speaker_independent_tracked_v1"
    if output.is_relative_to(production) or output.is_relative_to(training_root / "src"):
        raise ValueError("Benchmark output must be separate from immutable production/source")
    if any((output / name).exists() for name in
           ("fold1_five_model_efficiency.json", "fold1_five_model_efficiency.csv", "integrity_before.json")):
        raise FileExistsError("Existing benchmark evidence must not be overwritten")
    environment = json.loads((training_root / "outputs/fold1_tracked_prelaunch/packages.json").read_text())
    require_environment(environment)
    if not torch.cuda.is_available() or "RTX 5080 Laptop" not in torch.cuda.get_device_name():
        raise RuntimeError("STOP: expected CUDA RTX 5080 Laptop GPU")
    from train import build_model
    from revision.protocol import expected_run_config
    configs = {method: json.loads((production / "fold_01" / method / "config.json").read_text())
               for method in METHODS}
    revision = configs["audio"]["revision"]
    training_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=training_root, text=True).strip()
    if training_sha != revision["git_commit"] or source_fingerprint(training_root) != revision["source_fingerprint"]:
        raise RuntimeError("STOP: training source provenance mismatch")
    if source_fingerprint(ROOT) != revision["source_fingerprint"]:
        raise RuntimeError("STOP: benchmark model source differs from training source")
    for method, config in configs.items():
        if config["revision"] != revision or config != expected_run_config(revision, method):
            raise ValueError(f"STOP: persisted canonical config mismatch: {method}")
        if revision["split"]["fold"] != 1 or config["frames"] != 8:
            raise ValueError("Expected production Fold 1 / 8 video frames")
    output.mkdir(parents=True, exist_ok=True)
    print("Checking immutable production artifacts (including all 30 weights)...", flush=True)
    before = artifact_snapshot(production)
    (output / "integrity_before.json").write_text(json.dumps(before, indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20261010)
    waveform = torch.randn(WAVEFORM_SHAPE, device="cuda", dtype=torch.float32)
    video = torch.randn(VIDEO_SHAPE, device="cuda", dtype=torch.float32)
    import torchinfo
    profiler_sources = {p.name: sha256(p) for p in sorted(Path(torchinfo.__file__).parent.glob("*.py"))}
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_sha": training_sha, "benchmark_base_git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_script_sha256": sha256(Path(__file__)), "source_fingerprint": source_fingerprint(ROOT),
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(), "precision": "FP32", "tf32": False,
        "autocast": False, "cudnn_benchmark": False, "batch_size": 1,
        "waveform_shape": list(WAVEFORM_SHAPE), "video_shape": list(VIDEO_SHAPE),
        "input_seed": 20261010, "warmup_per_repeat": 20, "timed_passes_per_repeat": 100,
        "repeats": 3, "package_snapshot": environment, "torchinfo_version": torchinfo.__version__,
        "torchinfo_source_sha256": profiler_sources, "operation_definition": OPERATION_DEFINITION,
        "caption": CAPTION, "cpu_threads": torch.get_num_threads(),
        "wall_clock": "CLOCK_MONOTONIC_RAW (unadjusted); perf_counter stored for diagnosis only",
        "nvidia_smi_before": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()}
    print(json.dumps({k: metadata[k] for k in ("device", "torch_version", "cuda_version", "precision")}), flush=True)
    rows = []
    for method in METHODS:
        print(f"Benchmarking {method}", flush=True)
        path = production / "fold_01" / method / "best.pt"
        checkpoint_hash = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = restore_model(configs[method], checkpoint, build_model).to(device="cuda", dtype=torch.float32).eval()
        del checkpoint
        inputs = select_inputs(method, waveform, video)
        with torch.inference_mode():
            logits = model(*inputs)
            if logits.shape != (1, 8) or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid logits shape/finiteness: {method}")
        count = total_parameters(model)
        latency = measure_latency(model, inputs, 20, 100, 3)
        profile = operation_profile(model, inputs, output, method)
        rows.append({"method": method, "checkpoint": str(path), "checkpoint_sha256": checkpoint_hash,
                     "config_sha256": sha256(path.parent / "config.json"), "config": configs[method],
                     "build_model_kwargs": model_kwargs(configs[method], build_model), "strict_load": True,
                     "input_shapes": [list(item.shape) for item in inputs], "output_shape": list(logits.shape),
                     "parameters": count, "parameters_millions": count / 1e6,
                     "profile": profile, "latency": latency})
        print(f"  params {count:,}; mult-adds {profile['total_mult_adds']}; "
              f"latency {latency['combined']['mean_ms']:.3f} ms", flush=True)
        del model, logits, inputs
        gc.collect()
        torch.cuda.empty_cache()
    print("Rechecking all production hashes/mtimes and package/source provenance...", flush=True)
    after = artifact_snapshot(production)
    (output / "integrity_after.json").write_text(json.dumps(after, indent=2) + "\n")
    require_environment(environment)
    if after != before or source_fingerprint(training_root) != revision["source_fingerprint"]:
        raise RuntimeError("STOP: immutable production/source changed during benchmark")
    metadata["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    payload = {"schema": "spmb-five-model-efficiency-v1", "metadata": metadata, "models": rows,
               "integrity": {"production_files_and_symlinks": len(before), "before_equals_after": True,
                             "all_30_runs_untouched": True, "package_snapshot_exact_match": True}}
    with (output / "fold1_five_model_efficiency.json").open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    fields = ["method", "parameters", "parameters_millions", "total_mult_adds", "estimated_gmacs",
              "latency_mean_ms", "latency_median_ms", "latency_sample_stdev_ms", "wall_mean_ms",
              "training_git_sha", "checkpoint_sha256", "device", "precision", "torch_version",
              "cuda_version", "torchinfo_version", "input_shapes", "profile_status", "profile_errors"]
    with (output / "fold1_five_model_efficiency.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": row["method"], "parameters": row["parameters"],
                             "parameters_millions": row["parameters_millions"],
                             "total_mult_adds": row["profile"]["total_mult_adds"],
                             "estimated_gmacs": row["profile"]["estimated_gmacs"],
                             **{f"latency_{key}": row["latency"]["combined"][key]
                                for key in ("mean_ms", "median_ms", "sample_stdev_ms")},
                             "wall_mean_ms": row["latency"]["wall_repeat_average_statistics"]["mean_ms"],
                             **{key: metadata[key] for key in ("training_git_sha", "device", "precision", "torch_version", "cuda_version", "torchinfo_version")},
                             "checkpoint_sha256": row["checkpoint_sha256"], "input_shapes": json.dumps(row["input_shapes"]),
                             "profile_status": row["profile"]["status"], "profile_errors": json.dumps(row["profile"].get("error_classes", []))})
    print("PASS: five actual best checkpoints benchmarked; all production bytes/mtimes unchanged.", flush=True)


if __name__ == "__main__":
    main()
