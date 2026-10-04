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
there is no incomplete-production override in this PR. The local `data/` folder
currently has no RAVDESS pairs, so real-data completeness remains unverified.
The audit is structural; it does not decode all files or certify media quality.

## Cheap end-to-end smoke

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --smoke --fold 1 --model chumachenko_ia \
  --output-root outputs/revision_smoke_pr
```

Use a fresh output root for each rerun. This creates decodable synthetic media
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
  --output-root outputs/speaker_independent --epochs 20 --batch-size 4
```

Full six-fold suite, in audio/video/concat/IA/gated/xattn order for each fold:

```bash
PYTHONPATH=src .venv/bin/python -m revision.run \
  --data-root /absolute/path/to/RAVDESS --model all \
  --output-root outputs/speaker_independent --epochs 20 --batch-size 4
```

All methods use the fixed 16/4/4 disjoint actor partitions in `SPEC.md`.
Every fusion method loads only the audio/video best checkpoints in its own fold.
Old random-split checkpoints, checkpoints from other folds, and mismatched
provenance are forbidden. There are no arbitrary checkpoint CLI inputs.
Pretrained `microsoft/wavlm-base` and ImageNet ResNet18 are required for production.
Unavailable pretrained WavLM weights cause an error rather than random training.

The runner's current optimization policy is the existing trainer's single-stage
Adam defaults: LR 1e-3, weight decay 1e-4, WavLM backbone frozen, mean temporal
pooling, no alignment/emotion-prior additions, early stopping patience 10. Fusion
methods warm-start from same-fold unimodal checkpoints. This is a shared initial
protocol, not a completed hyperparameter selection exercise. Review the policy
before launching production; this PR does not run or optimize the final matrix.

## Artifacts and reproducibility

Each model writes `fold_NN/MODEL/{best.pt,config.json,metrics.json,pairs.csv}`.
Each fold writes `split.json` with exact actor assignments and provenance.
The root stores audit reports, a smoke/production mode sentinel, and aggregate
JSON/CSV. Configs include every trainer argument, seed, encoder selection,
preprocessing settings, library versions, Git commit and source fingerprint.
The manifest fingerprint includes filenames, size and mtime; preserve these when
re-evaluating copied datasets. Full file-content hashing is not provided.

Existing selected model runs are refused before training. Completed audio/video
prerequisites may be reused for a new fusion model only when fold, dataset,
source, settings and mode match exactly. There is no partial-training resume;
use a fresh output root after failures or code/config changes.

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
production preflight refusal, and the full orchestration schedule using a mock
trainer (no production training).

The pre-existing backend test
`TestStreamingEmotionSession.test_session_builds_sliding_window_and_updates_cadence`
expects 2 buffered frames but receives 4. It also fails unchanged on base commit
`d50f07c`. No backend file is changed or assertion weakened in this PR.

Standalone evaluation initializes encoder structure without downloading pretrained
weights, then strictly loads the complete saved model state. This supports offline
checkpoint evaluation while production training still requires pretrained weights.
