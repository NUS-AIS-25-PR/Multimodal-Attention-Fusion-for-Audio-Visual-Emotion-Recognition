# ACCEPTANCE.md

The first handoff implementation is considered complete only when all items below are satisfied.

## A. Fixed-fold correctness

- [x] Six fixed folds exist exactly as specified.
- [x] Each fold has 16 train actors, 4 validation actors, and 4 test actors.
- [x] No actor overlaps across partitions within a fold.
- [x] Actors 1..24 appear exactly once as test actors across all six folds.
- [x] Actors 1..24 appear exactly once as validation actors across all six folds.
- [x] Automated tests enforce these invariants.

## B. Dataset audit

- [x] There is a command or callable audit for the local RAVDESS paired dataset.
- [x] It reports per-actor audio/video/pair counts.
- [x] It reports per-actor and global class distributions.
- [x] It detects duplicate pair keys.
- [x] Production CV fails on incomplete data by default.
- [x] Synthetic-data tests cover success and failure cases.

## C. Checkpoint integrity

- [x] Fold outputs do not overwrite other folds.
- [x] Each fold saves its split metadata.
- [x] Each model saves enough config to identify the experiment.
- [x] Test evaluation reloads the best validation checkpoint.
- [x] A test verifies the best-checkpoint path is actually used.

## D. Leakage prevention

- [x] The runner never points multimodal models to old random-split unimodal checkpoints.
- [x] Fold N multimodal models use Fold N unimodal checkpoints.
- [x] Documentation explicitly warns against cross-fold or old-checkpoint reuse.
- [x] Smoke logs make the source checkpoint paths visible.

## E. Evaluation consistency

- [x] Standalone evaluation respects checkpoint `use_wavlm` / preprocessing configuration.
- [x] WavLM models receive raw waveform inputs, not mel inputs.
- [x] All revision methods use the same fold actor definitions.
- [x] Accuracy, precision, recall, and macro-F1 are available.

## F. Chumachenko external baseline

- [x] A clearly named adapted Intermediate Attention baseline exists.
- [x] It uses one attention head.
- [x] It consumes current WavLM and ResNet18 representations.
- [x] Its forward pass produces `[batch, 8]` logits.
- [x] Its implementation contains bidirectional intermediate audio/video attention semantics rather than simple concatenation.
- [x] It can train/evaluate through the same revision runner.
- [x] Tests cover basic tensor/shape behavior.
- [x] Documentation calls it adapted/reimplemented, not an exact reproduction.

## G. Cross-validation orchestration

- [x] One fold can be selected explicitly.
- [x] One model can be selected explicitly.
- [x] The full six-fold configuration is supported.
- [x] Per-fold metrics are serialized.
- [x] Aggregate mean and standard deviation are generated.
- [x] Raw per-fold values remain available.
- [x] A cheap smoke mode or equivalent exists.

## H. First-PR scope control

- [x] No backend/frontend/Docker/ONNX refactor is included without necessity.
- [x] No new core paper architecture is introduced.
- [x] No final expensive six-fold production training is required for merge.
- [x] Revision, data, and attention tests pass. The unrelated backend streaming
  failure also occurs on the untouched base commit; see `REVISION_RUNS.md`.
- [x] PR description lists commands and evidence.

## H2. Production profiles and safe completed-run resume

- [x] Explicit canonical profiles preserve audio stage 2, Gated two-stage and
  XAttn gated head/d_model 96/smoothing 0.05/two-stage definitions.
- [x] IA/concat use Gated optimization; IA weights remain query-summed and
  unnormalized, without original softhard dropout.
- [x] Profile IDs and full resolved configs are persisted and validated across
  folds, checkpoints, metrics, warm starts and aggregation.
- [x] Complete matching runs are skipped without retraining/re-evaluation.
- [x] Incomplete, corrupt and mismatched existing runs fail before new training.
- [x] Production loader workers default to -1; smoke uses 0.
- [x] Automated tests cover canonical profiles and completed-fold resume.
- [x] No real RAVDESS training is started during this follow-up.

## I. Scientific acceptance condition for the later production run

The final experiment phase will be considered scientifically successful if:

- all six folds complete without leakage;
- all primary methods are evaluated on the same folds;
- the external adapted baseline is included;
- aggregate mean ± std is available;
- scores are traceable to saved best-validation checkpoints;
- a complete split/config/metrics trail exists for every number intended for the revised paper.

There is **no minimum accuracy threshold** required for acceptance of the engineering work.
