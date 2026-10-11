# Five-model checkpoint-only efficiency benchmark

This benchmark measures `audio`, `video`, `gated`, `xattn`, and adapted
`chumachenko_ia` together. It does not train, read dataset media, evaluate held-out
examples, calculate recognition metrics, or import `src/benchmark_table6.py`.
All models are built by the unchanged training `build_model` from their persisted
Fold 1 configs and strictly restored from their actual `best.pt` files.

The training source commit is `0ae967afc654d3234620862437d63f2f33ab57e7`.
Training source fingerprint is
`e0ac3ba8452d283be16cda83add02069020b5e917f0e760eb79e134c07da9f6b`.
The harness requires that source and the exact 111-package Fold 1 snapshot,
including MediaPipe 0.10.21 and torchinfo 1.8.0. It stops on mismatch; do not
install or resynchronize packages to work around a failed guard. The bootstrap
flags `checkpoint_init=True` and `pretrained_video=False` prevent downloads;
strict restoration replaces the complete bootstrap state before any inference.

## Command actually executed

From the immutable training checkout, using the separate benchmark worktree:

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
PYTHONPATH=/home/louis/projects/MultimodalEmotionRecognition-efficiency/src \
.venv/bin/python -u \
  /home/louis/projects/MultimodalEmotionRecognition-efficiency/scripts/benchmark_revision_efficiency.py \
  --training-root /home/louis/projects/MultimodalEmotionRecognition \
  --output-root /home/louis/projects/MultimodalEmotionRecognition/outputs/revision_efficiency \
  > outputs/revision_efficiency/benchmark.log 2>&1
```

The destination must be fresh. Existing result JSON/CSV/integrity evidence is
never overwritten. The script writes only into the benchmark destination, and
checks hashes/mtimes of all 493 production files/symlinks before and after,
including all 30 best checkpoints. The production output tree is unchanged.
For another measurement, use a new output directory and preserve this capture.
No `uv sync`, dependency upgrade, actor-split edit, model edit or training occurs.

## Measurement definition and results

NVIDIA GeForce RTX 5080 Laptop GPU; driver 616.56; Torch 2.10.0+cu130,
CUDA 13.0, cuDNN 91501; FP32, autocast/TF32 disabled. CPU thread count is 1.
Batch size is 1. Inputs are GPU-preallocated standard-normal tensors:
waveform `[1,48000]` and video `[1,8,3,112,112]`. Unimodal methods receive
their own modality only; fusion receives `(video, waveform)`.

| Method | Total parameters (M) | Estimated GMacs (torchinfo) | Inference time (ms/sample) |
|---|---:|---:|---:|
| Audio | 94.979 | 7.417 | 10.507 |
| Video | 11.181 | 3.879 | 3.663 |
| Gated | 106.621 | 11.296 | 14.048 |
| XAttn | 106.607 | 11.295 | 14.152 |
| Adapted Chumachenko IA | 106.391 | 11.296 | 14.157 |

Exact integers and unrounded latency values are in the JSON/CSV capture.
Parameter count is `sum(p.numel() for p in model.parameters())`, including all
registered frozen/unused encoder and classifier heads. Latency is measured;
the operation column is a partial theoretical estimate, not a timing-derived
count and not a rigorous FLOP estimate. All five use the same harness and tool.

### Exact proposed Table 1 caption

> Total parameters include all registered encoder and classifier parameters,
> including frozen and unused registered heads. Estimated GMacs are torchinfo
> 1.8.0 total_mult_adds divided by 10^9, not rigorous GFLOPs; sequence Linear
> and dynamic attention operations are under-counted. Latency is the measured
> mean CUDA-event elapsed time per batch-1 forward on an NVIDIA GeForce RTX
> 5080 Laptop GPU, in eval() and torch.inference_mode(), FP32 with TF32 and
> autocast disabled. Each model uses its strictly restored Fold 1 best.pt,
> with a preallocated random waveform [1,48000] and/or video [1,8,3,112,112].
> Three repeats each follow 20 warmups and contain 100 timed forwards; CUDA
> is synchronized before and after each timed loop; unadjusted Linux
> CLOCK_MONOTONIC_RAW wall time is retained as a cross-check. Disk I/O, model loading,
> data loading, face cropping and audio resampling are excluded. No held-out
> data or recognition metrics enter this efficiency measurement.

The JSON stores all 300 individual CUDA-event samples/model, each repeat's
statistics, pooled sample mean/median/sample SD, repeat-mean statistics, raw
wall-loop times and per-repeat GPU temperature/power/clocks. These are forward
times rather than preprocessing-inclusive application response times.
The three fusion timing differences are small relative to measurement
variability; these measurements do not establish a significant speed advantage.
Laptop clocks/power are observed rather than locked.

### Operation-count coverage

The raw `total_mult_adds` are 7,417,048,936 (Audio), 3,878,861,832 (Video),
11,295,771,753 (Gated), 11,295,462,089 (XAttn) and 11,295,541,608 (IA).
No multiplication by two converts these estimates to FLOPs.

[torchinfo's documentation](https://github.com/TylerYep/torchinfo/tree/v1.8.0)
labels its result Mult-Adds. We inspected and hash-recorded the installed 1.8.0
implementation: `LayerInfo.calculate_macs` uses parameter/output heuristics and
`ModelStatistics` sums leaf-module counts. Convolution counts include bias;
normalization/other parameterized layers receive heuristics rather than an
operator trace. Sequence `Linear` uses batch size, omitting multiplication by
token length. Functional QK/AV matmuls, softmax, positional-bias logic,
reductions and elementwise fusion are not reliably covered; non-leaf
MultiheadAttention is also incompletely represented. Per-layer coverage and
full summaries are preserved. Similar estimated fusion counts do not establish
equal true computational cost, especially for XAttn/IA. If profiling fails,
that method records N/A and the exact exception chain/trace; there is no fallback
profiler or fabricated number. All five profiles succeeded in this capture.

IA is the project's eight-class WavLM/ResNet18 adaptation. The original ICPR
2022 study used seven RAVDESS categories and does not supply these efficiency
values for this adapted model. None of its recognition or complexity numbers
are imported here.

### Clock diagnosis

An initial uniform five-model capture also retained ordinary `perf_counter`
wall time. It was approximately 9% shorter than CUDA-event duration. A separate
100-matmul clock probe showed that `perf_counter`, adjusted monotonic and Unix
clocks agree, while `CLOCK_MONOTONIC_RAW` agrees with the CUDA clock. We did not
change clock/environment settings. All five were recaptured uniformly with raw
wall-clock cross-checks. Final CUDA-event/raw wall mean times differ by less
than 0.13%; ordinary `perf_counter` values are diagnostic-only.
Initial measurements and the probe remain at
`outputs/revision_efficiency_clock_diagnostic/` locally; they are excluded from
the final table and retained in the PR's separate diagnostic folder.

## Artifacts and validation

Required local files:

- `outputs/revision_efficiency/fold1_five_model_efficiency.json`
- `outputs/revision_efficiency/fold1_five_model_efficiency.csv`
- `outputs/revision_efficiency/benchmark.log`
- `outputs/revision_efficiency/integrity_before.json` and `integrity_after.json`
- `outputs/revision_efficiency/*_torchinfo.txt` and `unit_tests.log`

The reviewable capture is under
[`results/ieee_spmb_2026/fold1_efficiency/`](../results/ieee_spmb_2026/fold1_efficiency/README.md),
separate from the previously archived six-fold result evidence.
JSON records checkpoint/config/script/profiler hashes, complete configs,
Git provenance, hardware/software/precision, shapes and full package snapshot.

```bash
OMP_NUM_THREADS=1 /home/louis/projects/MultimodalEmotionRecognition/.venv/bin/python \
  -m unittest discover -s tests -p test_revision_efficiency.py -q
```

Nine CPU-only contracts passed, including config/strict restoration, modality
shapes/order, frozen parameter counting, insufficient sampling rejection,
profiler failure, environment mismatch, tamper detection and separate clock
statistics. The actual GPU benchmark strictly restored all five checkpoints,
checked finite `[1,8]` logits on random inputs, and confirmed all production
bytes/mtimes and the exact environment snapshot remained unchanged.
