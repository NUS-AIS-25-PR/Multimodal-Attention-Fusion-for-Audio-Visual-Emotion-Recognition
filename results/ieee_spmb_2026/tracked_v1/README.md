# IEEE SPMB 2026: complete actor-independent production results

This is the immutable publication snapshot for five methods × six folds (30
completed runs), trained on the complete 1440-pair RAVDESS speech subset. Training
and test inference are finished. No additional training was launched for this PR.

Training source commit: `0ae967afc654d3234620862437d63f2f33ab57e7`.
The archive commit is separate from that scientific provenance. All copied files
retain their original bytes, configuration, absolute paths, timestamps in reports,
and checkpoint-selection evidence. Do not edit them to make paths portable.

## Six-fold results

Percentages; mean ± sample standard deviation (`ddof=1`). Precision and recall
use the eight-class macro average, with zero contribution for an undefined class.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Audio | 65.90 ± 9.52 | 69.99 ± 6.51 | 65.89 ± 9.31 | 64.58 ± 9.96 |
| Video | 56.04 ± 7.31 | 56.48 ± 8.85 | 54.75 ± 7.17 | 53.22 ± 8.64 |
| Gated | 70.28 ± 7.18 | 75.66 ± 4.84 | 69.47 ± 6.61 | 68.67 ± 7.42 |
| Adapted IA | 56.25 ± 3.96 | 60.59 ± 4.91 | 55.34 ± 3.40 | 53.95 ± 4.19 |
| XAttn | 68.75 ± 6.45 | 72.53 ± 6.75 | 68.03 ± 5.94 | 67.23 ± 6.20 |

All 30 per-fold Accuracy/Precision/Recall/Macro-F1 values, best epochs, validation
Macro-F1, early-stopping flags and process timings are in
[per_fold_metrics.csv](outputs/speaker_independent_tracked_v1/per_fold_metrics.csv).
The table source is [mean_std_table.csv](outputs/speaker_independent_tracked_v1/mean_std_table.csv);
[aggregate.json](outputs/speaker_independent_tracked_v1/aggregate.json) preserves
raw fold scores and common provenance. [six_fold_report.json](outputs/speaker_independent_tracked_v1/six_fold_report.json)
records total active command runtime and historical interruptions.

## Per-fold test results

Percentages. These are held-out scores from each run's single test evaluation
with its best validation Macro-F1 checkpoint, not scores from the last epoch.

| Method | Fold | Accuracy | Macro precision | Macro recall | Macro-F1 | Best epoch | Validation Macro-F1 | Epochs / early stop |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Audio | 1 | 77.08 | 77.45 | 77.73 | 76.51 | 14 | 76.32 | 20 / false |
| Audio | 2 | 75.83 | 78.38 | 73.83 | 74.12 | 18 | 68.02 | 20 / false |
| Audio | 3 | 57.50 | 65.27 | 56.25 | 56.95 | 11 | 63.67 | 20 / false |
| Audio | 4 | 53.33 | 63.20 | 54.69 | 50.37 | 10 | 67.42 | 20 / true |
| Audio | 5 | 65.00 | 65.84 | 64.06 | 63.54 | 10 | 71.39 | 20 / true |
| Audio | 6 | 66.67 | 69.77 | 68.75 | 65.95 | 15 | 76.94 | 20 / false |
| Video | 1 | 55.83 | 51.97 | 52.34 | 50.92 | 11 | 63.88 | 20 / false |
| Video | 2 | 64.17 | 65.99 | 62.11 | 62.76 | 18 | 57.84 | 20 / false |
| Video | 3 | 65.00 | 67.21 | 64.45 | 64.18 | 11 | 57.40 | 20 / false |
| Video | 4 | 52.50 | 50.27 | 52.34 | 48.11 | 10 | 49.37 | 20 / true |
| Video | 5 | 46.25 | 45.34 | 45.31 | 41.92 | 11 | 48.33 | 20 / false |
| Video | 6 | 52.50 | 58.13 | 51.95 | 51.45 | 19 | 60.69 | 20 / false |
| Gated | 1 | 66.25 | 76.25 | 62.50 | 61.39 | 18 | 74.58 | 26 / true |
| Gated | 2 | 81.25 | 81.69 | 78.52 | 78.63 | 22 | 74.25 | 30 / true |
| Gated | 3 | 77.08 | 80.24 | 76.56 | 77.11 | 25 | 73.08 | 30 / false |
| Gated | 4 | 67.08 | 69.52 | 68.75 | 66.01 | 19 | 71.29 | 27 / true |
| Gated | 5 | 67.08 | 75.22 | 66.02 | 66.56 | 22 | 65.78 | 30 / true |
| Gated | 6 | 62.92 | 71.04 | 64.45 | 62.33 | 11 | 70.37 | 19 / true |
| Adapted IA | 1 | 54.17 | 56.20 | 50.78 | 48.76 | 22 | 63.25 | 30 / true |
| Adapted IA | 2 | 62.08 | 66.05 | 60.16 | 60.07 | 12 | 58.87 | 20 / true |
| Adapted IA | 3 | 60.00 | 67.42 | 57.81 | 57.63 | 10 | 65.26 | 18 / true |
| Adapted IA | 4 | 55.83 | 58.83 | 56.25 | 53.43 | 2 | 57.40 | 10 / true |
| Adapted IA | 5 | 53.33 | 58.72 | 53.12 | 52.61 | 15 | 52.83 | 23 / true |
| Adapted IA | 6 | 52.08 | 56.33 | 53.91 | 51.18 | 12 | 66.15 | 20 / true |
| XAttn | 1 | 71.25 | 79.92 | 67.58 | 66.96 | 32 | 77.62 | 35 / false |
| XAttn | 2 | 72.92 | 71.87 | 70.70 | 70.49 | 11 | 68.32 | 21 / true |
| XAttn | 3 | 75.42 | 78.51 | 75.00 | 74.99 | 14 | 73.37 | 24 / true |
| XAttn | 4 | 71.25 | 75.60 | 72.66 | 70.69 | 31 | 71.52 | 35 / false |
| XAttn | 5 | 62.92 | 65.80 | 62.50 | 62.16 | 31 | 68.27 | 35 / false |
| XAttn | 6 | 58.75 | 63.46 | 59.77 | 58.07 | 1 | 67.56 | 11 / true |

## Protocol and preserved configuration

- Six fixed groups: actors 01–04, 05–08, 09–12, 13–16, 17–20, 21–24.
  Fold N tests on group N, validates on the next cyclic group, and trains on the
  remaining 16 actors. Train/validation/test actors are disjoint; each fold has
  960/240/240 pairs. Each actor appears once in test and validation.
- WavLM-base and ImageNet-pretrained ResNet18 are unchanged. Canonical profiles,
  profile IDs and resolved configs are preserved in each run. No old random-split
  or cross-fold checkpoint is reused. Fusion warm starts use only that fold's
  actor-independent Audio/Video checkpoints. Concat was not run.
- Adapted IA is the ICPR 2022 one-head, query-summed Intermediate Attention
  mechanism using shared current encoders, without weight normalization or the
  original softhard dropout. It is not an exact reproduction of the entire
  published MFCC/EfficientFace pipeline or its original folds.
- W&B remained offline: project `ieee-spmb-2026`, group
  `spmb2026-canonical-tracked-v1`. Run configuration/history/predictions are
  authoritative local artifacts. The SDK bundles are retained for provenance;
  this PR does not upload them to W&B and does not contain API credentials.

## Paper figures

[Figure directory](outputs/paper_figures_six_fold_tracked_v1) contains 66 distinct
figures, each exported as vector PDF/SVG and 300 DPI PNG (198 exports): 30 learning
curves, 30 eight-class confusion matrices, five pooled confusion matrices, and
one four-metric model comparison with sample-SD error bars.

- [Model comparison PDF](outputs/paper_figures_six_fold_tracked_v1/model_comparison.pdf)
- [Model comparison PNG](outputs/paper_figures_six_fold_tracked_v1/model_comparison.png)
- Pooled confusion PDFs: [Audio](outputs/paper_figures_six_fold_tracked_v1/audio_pooled_confusion.pdf),
  [Video](outputs/paper_figures_six_fold_tracked_v1/video_pooled_confusion.pdf),
  [Gated](outputs/paper_figures_six_fold_tracked_v1/gated_pooled_confusion.pdf),
  [IA](outputs/paper_figures_six_fold_tracked_v1/chumachenko_ia_pooled_confusion.pdf),
  [XAttn](outputs/paper_figures_six_fold_tracked_v1/xattn_pooled_confusion.pdf).

Every method pools exactly 1440 unique held-out predictions before row
normalization. All plots have 6/6 coverage; no INCOMPLETE or smoke labels.
The original [figure manifest](outputs/paper_figures_six_fold_tracked_v1/figures_manifest.json)
retains input hashes and original generation paths. No synthetic paper values,
raw RAVDESS media, cropped face images or synthetic smoke figures are published.

## Artifact inventory and model weights

[artifact_inventory.json](artifact_inventory.json) lists the size and SHA256 of
630 unchanged copied artifacts plus all 31 retained-local checkpoints: 30
completed best-validation checkpoints and one interrupted Fold 6 Video
checkpoint that was not used for any final result. All other original local
artifacts remain preserved. Per user instruction, approximately 9.56 GiB of
checkpoint binaries remain locally stored and are not committed to Git.

The archive mirrors `outputs/` under this directory. It includes all final run
CSVs/JSONs, pair manifests, offline W&B records, all paper figures, original
supervisor scripts/logs/timings/GPU samples, prelaunch package/audit snapshots,
and both interrupted-run metadata/logs. Selected real-media preflight reports
are in [preflight/](preflight/); media-derived visualizations remain local.
Symlinks are represented in the inventory, with their canonical targets copied
once. Scoped `.gitattributes` prevent line-ending conversion and whitespace lint
on immutable captured evidence; authored documentation and verification code
retain normal formatting checks. Nothing from unrelated dirty files in the development worktree is included.

This is a results snapshot, not a relocated resumable training directory:
`revision.run`, `revision.aggregate`, and `revision.figures` deliberately demand
complete matching checkpoint/provenance paths. Their original commands and
paths are retained. Use the standalone verifier below on a fresh checkout;
recreating figures through the production CLI additionally needs the original
checkpoint binaries and matching paths/environment. No checkpoint-free result
is presented as a new model evaluation.

## Verification

From the repository root, with Python 3.10+ and no extra dependencies:

```bash
python3 results/ieee_spmb_2026/tracked_v1/verify_archive.py
```

The verifier checks every copied artifact hash, fixed splits, 30 consistent
config/history/prediction records, the first best validation epoch, exclusive
held-out sample coverage, confusion counts, recomputed test metrics, mean and
sample SD, and all 198 exports. It uses saved predictions, not model inference.
On the original Ubuntu machine, also verify all retained checkpoint hashes:

```bash
python3 results/ieee_spmb_2026/tracked_v1/verify_archive.py --check-local-checkpoints
```

The optional hash check proves local file identity; full checkpoint loading and
existing production completion validation were also executed against all 30
original runs before archiving. See [validation.json](validation.json) for
executed checks, including three archive contracts (complete snapshot, byte tampering, and semantic
tampering even with an updated hash) and 35 focused regression tests. Run the
archive contracts without installing project dependencies:

```bash
python3 -m unittest discover -s results/ieee_spmb_2026/tracked_v1 -p test_archive_integrity.py -v
```

## Runtime, interruptions, and review notes

Active training command time totals 42,537.53 seconds (11 h 48 min 58 s), including
validation, loading, test, safe-skip checks and failed attempts. It excludes idle
waiting between sessions and the final figure generation (about 97 seconds).
Production code, hyperparameters and tracking settings were unchanged.

Two interrupted attempts are preserved separately and excluded from aggregates:

1. Fold 2 Gated initialization failed at zero epochs following a Hugging Face
   SSL EOF/closed HTTP client. Its directory was archived before a clean run.
2. Fold 6 Video stopped after 13 epochs when the dependency guard detected six
   packages changed from the Fold 1 snapshot. The environment was restored,
   the interrupted directory/checkpoint archived, and this model restarted from
   epoch 1, without partial optimizer recovery. A computer restart followed the
   guard stop. The final completed Video run selected epoch 19.

All 111 distribution versions match the Fold 1 snapshot; in particular MediaPipe
0.10.21, NumPy 1.26.4, protobuf 4.25.9 and all three OpenCV packages 4.11.0.86.
The project lockfile has newer conflicting versions. Running `uv run office-run
codex` in the training project can sync those six packages; use the installed
`office-run codex` or `uv run --no-sync office-run codex` to avoid that synchronization.

No fatal NaN/OOM was recorded. Benign headless EGL/DRI3/llvmpipe, TFLite feedback,
offline-W&B resume and unauthenticated Hugging Face warnings occurred. The loader
swallows some face-detection exceptions; logs do not prove zero detector errors
across every production clip. Real-media preflight found 48/48 detections,
368 sampled cropped frames, zero crop/full-frame fallback errors and 16 padded
repeated frames. This sampled check does not certify historical paper cropping.

IA validation loss was high in some folds. IA Fold 4 selected epoch 2; XAttn
Fold 6 selected epoch 1 and early-stopped at epoch 11. These are valid protocol
outcomes that deserve discussion, not reasons to tune retrospectively or select
checkpoints using test results. Gated has the highest aggregate Macro-F1 in this
run; error bars alone do not establish a statistically significant advantage.
