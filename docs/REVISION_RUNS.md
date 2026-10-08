# IEEE SPMB 2026 revision experiments

Use the existing modern environment; no legacy PyTorch installation is needed.
The first PR supplies infrastructure and a cheap synthetic validation path.
Production training must wait for PR review.

## Local completeness preflight

```bash
PYTHONPATH=src .venv/bin/python -m revision.audit \
  --data-root /absolute/path/to/RAVDESS --output outputs/ravdess_audit.json
```

This audits modality `02` MP4 + modality `03` WAV speech pairs by filename.
It reports discovered actors, audio/video/pair counts and class counts for every
actor, global counts, duplicate keys with source paths, missing counterparts,
invalid filenames, and missing expected keys. Production requires all 1440
unique designed pairs (24 actors × 60 samples), including the exact neutral
intensity/statement/repetition composition. Incomplete data raises an error;
there is no incomplete-production override. The local dataset was organized
under `data/Actor_01`–`data/Actor_24` and audited on October 8, 2026: 24 actors ×
60 pairs = 1440, complete=True. Rerun preflight for each experiment. Git does
not include media, so a fresh checkout must supply its own dataset.
The audit is structural; it does not decode all files or certify media quality.

## Local tracking and optional W&B

Every model/fold always saves `history.csv`, `test_predictions.csv`,
`confusion_matrix.json` and `tracking.json`. History uses eight-class macro
metrics, training stage and optimizer-group LRs actually used in the epoch.
Test predictions come from the single best-checkpoint test pass. Resume
validates all local evidence, and completed skips leave bytes/mtimes unchanged
without opening SDK runs, appending history or evaluating tests again.

W&B is disabled by default. Authenticate interactively for online operation:

```bash
.venv/bin/wandb login
```

Credentials belong in the client's local credential store, never this repo.
Do not put keys in source, shell arguments, PRs or logs. After review only:

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --data-root data --fold 1 --model gated \
  --output-root outputs/speaker_independent_tracking \
  --wandb-mode online --wandb-project ieee-spmb-2026 \
  --wandb-group spmb2026-canonical-v1
```

Use `--wandb-entity TEAM` if needed. Default group combines commit/dataset/root;
names identify model/fold. Transport settings do not change training profiles
and may change on a completed resume. SDK calls preserve RNG state. If W&B is
unavailable, only the exception type is reported and local logging continues.
Online initialization times out after 30 seconds. Local success does not prove
remote upload success; inspect `tracking.json` and W&B separately.

Offline smoke requires no login or remote service:

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --smoke --fold 1 --model chumachenko_ia \
  --output-root outputs/revision_tracking_smoke --wandb-mode offline
```

Optional later upload of an offline bundle (requires login):

```bash
.venv/bin/wandb sync outputs/revision_tracking_smoke/fold_01/audio/wandb/offline-run-*
```

The harness sends scalars/config only. It does not upload raw audio/video,
source code or prediction files. Disable tracking with `--wandb-mode disabled`.

## Publication figures from local artifacts

No W&B, network connection or another evaluation pass is needed:

```bash
PYTHONPATH=src .venv/bin/python -m revision.figures \
  --input-root outputs/speaker_independent_tracking \
  --output-dir outputs/paper_figures
```

Repeat `--model gated --model xattn` to select methods. Outputs include learning
curves, eight-class row-normalized confusion matrices, mean ± sample-SD model
comparisons and pooled held-out matrices. Each exports PDF/SVG/300 DPI PNG;
a manifest records input hashes, coverage, statistics and pooled raw counts.
Pool counts before normalization; repeated held-out actors/keys are rejected.
Training/validation predictions never enter pooled matrices.

Missing/incomplete/mixed artifacts fail rather than invent values. Absent folds
are labeled INCOMPLETE with actual n/6 coverage. One fold has no SD error bar.
Partial results cannot replace the main six-fold table. Rows with no observations
display an em dash. Input artifacts are read without modification.

Synthetic previews require an explicit flag and are marked NOT FOR PAPER:

```bash
PYTHONPATH=src .venv/bin/python -m revision.figures \
  --input-root outputs/revision_tracking_smoke \
  --output-dir outputs/revision_tracking_smoke_figures --allow-smoke
```

Use a fresh production output root after review. Pre-tracking/interrupted
artifacts lack the new source/schema and cannot be silently augmented by
invented history or another test pass. Preserve older artifacts.

Tracking/figure unit checks:

```bash
PYTHONPATH=src:tests OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  test_revision test_tracking_figures test_data_services test_attention_integration -q
```

These 32 relevant checks pass. Full discovery executes 40 checks, with 39 passing
and the existing backend sliding-window failure documented below. Tests use
isolated fixtures; actual offline W&B smoke is separate from mocked online tests.

## Cheap end-to-end smoke

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --smoke --fold 1 --model chumachenko_ia \
  --output-root outputs/revision_smoke_pr
```

Use a fresh output root when source/config changes. An identical rerun skips
complete matching runs. This creates decodable synthetic media
for all 24 actors, one sample each. Fold 1 uses train actors 09–24, validation
05–08, and test 01–04. The production fold definition is unchanged; the smoke
exception applies only to data completeness and pretrained encoder size/weights.
It executes real dataset decoding, loaders, forward/backward, best-checkpoint
saving/reloading, test scoring, and serialization. Selecting fusion also runs
its fold-local audio/video prerequisites. WavLM is a tiny random configuration;
ResNet18 is the existing encoder with pretrained weights disabled. Smoke scores
are infrastructure checks and must never enter the revised paper.

Check the standalone evaluator against that same saved model:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python src/eval.py \
  --data_root outputs/revision_smoke_pr/synthetic_media \
  --checkpoint outputs/revision_smoke_pr/fold_01/chumachenko_ia/best.pt \
  --num_workers 0
```

The evaluator obtains waveform/mel selection, frame count, face-crop setting,
model architecture and test actors from the checkpoint. Conflicting explicit
arguments and a changed revision dataset manifest are rejected.

## Production commands, after review only

One selected model and fold:

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --data-root /absolute/path/to/RAVDESS --fold 1 --model chumachenko_ia \
  --output-root outputs/speaker_independent
```

Full six-fold suite, in audio/video/concat/IA/gated/xattn order for each fold:

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --data-root /absolute/path/to/RAVDESS --model all \
  --output-root outputs/speaker_independent
```

All methods use the fixed 16/4/4 disjoint actor partitions in `SPEC.md`.
Every fusion method loads only the audio/video best checkpoints in its own fold.
Old random-split checkpoints, checkpoints from other folds, and mismatched
provenance are forbidden. There are no arbitrary checkpoint CLI inputs.
Pretrained `microsoft/wavlm-base` and ImageNet ResNet18 are required for production.
Unavailable pretrained WavLM weights cause an error rather than random training.

Production uses canonical per-model profiles from `src/revision/profiles.py`.
Generic `--epochs`, `--batch-size` and `--lr` overrides are not accepted by this
runner. Full default resolved configs and IDs are in
[`REVISION_PROFILES.json`](REVISION_PROFILES.json). Shared runtime defaults are
seed 42, 8 frames, face crop enabled and `num_workers=-1` (existing auto policy).

| Model / profile ID (`spmb2026-<model>-v1`) | Epochs | Batch | LR | Weight decay | Patience |
|---|---:|---:|---:|---:|---:|
| audio | 20 | 16 | 1e-3 | 1e-4 | 10 |
| video | 20 | 16 | 1e-3 | 1e-4 | 10 |
| gated | 30 | 8 | 3e-4 | 1e-4 | 8 |
| concat | 30 | 8 | 3e-4 | 1e-4 | 8 |
| chumachenko_ia | 30 | 8 | 3e-4 | 1e-4 | 8 |
| xattn | 35 | 8 | 2e-4 | 2e-4 | 10 |

All profiles enable cosine annealing. Audio uses `wavlm_stage=2` and
`backbone_lr=3e-5`. Gated/concat/IA use two-stage training with 5 stage-1 epochs,
audio/video backbone LR 1e-5, two unfrozen WavLM layers and one unfrozen video
block. XAttn keeps the paper's gated head, d_model 96, four heads, attention/drop
path probabilities 0.1, label smoothing 0.05, two-stage training with 6 stage-1
epochs, audio/video backbone LR 8e-6, two WavLM layers and one video block.
Mean pooling and the existing Gated/XAttn architectures remain unchanged.
IA remains one-head, query-summed and unnormalized; no softhard pipeline is added.
Production remains pending review; these commands must not be launched yet.

Smoke has distinct `spmb2026-<model>-smoke-v1` IDs. It preserves production
architecture/optimizer settings, including audio stage 2 and the XAttn gated
head/d_model/label smoothing, but uses two epochs, batch size 4, two frames,
workers 0, no face crop, tiny random WavLM and non-pretrained ResNet18. Fusion
stage 1 lasts one epoch so epoch 2 exercises the actual stage transition.

## Artifacts and reproducibility

Each model writes `fold_NN/MODEL/{best.pt,config.json,metrics.json,pairs.csv}`.
Each fold writes `split.json` with exact actor assignments and provenance.
The root stores audit reports, a smoke/production mode sentinel, and aggregate
JSON/CSV. Configs include every trainer argument, seed, encoder selection,
preprocessing settings, library versions, Git commit and source fingerprint.
The manifest fingerprint includes filenames, size and mtime; preserve these when
re-evaluating copied datasets. Full file-content hashing is not provided.

Rerunning the same command safely resumes completed runs: all existing fold/model
directories are checked before any new training. A run is skipped only if
`best.pt`, `metrics.json`, `config.json` and the fold's `split.json` exist, are
readable, and exactly match the expected provenance and entire trainer config.
Profile ID, resolved profile/catalog, source fingerprint, Git commit, package
versions, dataset manifest, split, settings and warm-start paths must agree.
Best epoch/validation score/checkpoint path and valid test metrics must agree too.
The catalog contains all per-model profiles in every fold's shared provenance;
aggregation rechecks complete artifacts and canonical profiles across folds.

Missing artifacts, corrupt checkpoints and any mismatch fail loudly before new
training. Completed selected models and their dependencies are skipped without
re-evaluation or overwriting; only absent runs are trained. Incomplete epoch runs
cannot resume. A source/config/data change requires a fresh output root.
Older artifacts from the initial generic single-stage harness are intentionally
incompatible with this canonical-profile revision.

Test scoring uses the checkpoint chosen by validation macro-F1 and runs once.
Precision/recall/F1 use all eight fixed labels and zero for undefined terms.
Aggregation preserves every raw fold score and reports mean and sample standard
deviation (`ddof=1`). Partial coverage is explicitly marked incomplete; a
single-fold standard deviation is null, not zero. Never select only the best fold.

## Adapted external baseline

Chumachenko, Iosifidis and Gabbouj, “Self-attention fusion for audiovisual emotion
recognition with incomplete data,” ICPR 2022:
[paper](https://arxiv.org/abs/2201.11095),
[authors' implementation at inspected commit](https://github.com/katerynaCh/multimodal-emotion-recognition/blob/65232ce53c91bf51cf7328d6138c941e3639d352/models/multimodalcnn.py),
[attention implementation](https://github.com/katerynaCh/multimodal-emotion-recognition/blob/65232ce53c91bf51cf7328d6138c941e3639d352/models/transformer_timm.py).

`ChumachenkoIntermediateAttentionFusion` preserves bidirectional, one-head
attention probabilities and query-summed temporal modulation of each modality.
WavLM/ResNet18 sequences are projected to 128 dimensions, modulated, mean-pooled,
concatenated, and classified. It is an adapted/reimplemented fusion mechanism,
not an exact reproduction of the original MFCC/EfficientFace, temporal CNN,
modality-dropout or five-fold pipeline. Unequal sequence lengths make the
published unnormalized query-sum scale asymmetric; this is deliberate and merits
scientific review before production. No published checkpoint or score is reused.

## Automated validation

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q \
  src/train.py src/eval.py src/revision src/models/chumachenko_ia.py \
  src/models/wavlm_audio.py tests/test_revision.py
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_revision.py'
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
```

The revision tests cover exact folds/coverage, leakage rejection, complete and
corrupted synthetic audits, fold paths and checkpoint provenance, actual best
checkpoint reload after a worse later epoch, evaluator preprocessing, eight-class
metrics, aggregate sample std/partial coverage/mixing, pretrained failure,
production preflight refusal, canonical production profiles/IDs, and complete
matching resume versus incomplete/mismatched/corrupt artifact refusal. Full
production orchestration and cross-fold profile consistency are tested with
controlled audit/trainer mocks; no production training is performed.

The pre-existing backend test
`TestStreamingEmotionSession.test_session_builds_sliding_window_and_updates_cadence`
expects 2 buffered frames but receives 4. It also fails unchanged on base commit
`d50f07c`. No backend file is changed or assertion weakened in this PR.

Standalone evaluation initializes encoder structure without downloading pretrained
weights, then strictly loads the complete saved model state. This supports offline
checkpoint evaluation while production training still requires pretrained weights.
