# Final five-model efficiency capture

See [measurement definitions, Table 1 and exact caption](../../../docs/EFFICIENCY_BENCHMARK.md).
This capture uses actual Fold 1 best checkpoints from training commit
`0ae967afc654d3234620862437d63f2f33ab57e7`.

Use only [`final/fold1_five_model_efficiency.csv`](final/fold1_five_model_efficiency.csv)
and [`final/fold1_five_model_efficiency.json`](final/fold1_five_model_efficiency.json)
for the paper. JSON includes full configs, all 300 timed samples per model,
repeat/combined statistics, checkpoint SHA256 and profiler coverage.
Parameters are exact counts; torchinfo GMacs are incomplete estimates;
CUDA-event forward latency is measured in FP32 at batch size 1 on the RTX
5080 Laptop GPU. No recognition values are imported or recalculated.

The final capture ran on 2026-10-10. All five models loaded strictly, all five
torchinfo estimates succeeded and nine lightweight tests passed. All 493
production files/symlinks (including all 30 weights) have identical hashes and
modification times before/after. The 111-package training environment and
training source fingerprint remain unchanged. No training or held-out inference
was run; checkpoint binaries stay local.

`final/` also preserves full logs, individual profiler summaries and the two
production integrity inventories. `diagnostic/` preserves the initial uniform
capture, original harness, logs and independent clock probe. It is **excluded
from the final paper table**. Adjusted Linux clocks differed by approximately
9% from CUDA events/raw monotonic time; the final harness records unadjusted
`CLOCK_MONOTONIC_RAW` as its wall-clock cross-check. No clock setting was changed.

[`artifact_inventory.json`](artifact_inventory.json) records original absolute
source paths, sizes and SHA256 for all 23 captured files. These copies preserve
their original bytes, including CSV line endings. The training/figure evidence
in `results/ieee_spmb_2026/tracked_v1/` is not modified by this branch.
