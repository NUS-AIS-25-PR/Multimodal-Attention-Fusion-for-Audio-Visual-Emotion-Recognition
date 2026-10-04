# SPEC.md

## 1. Repository context

Target repository:

`NUS-AIS-25-PR/Multimodal-Attention-Fusion-for-Audio-Visual-Emotion-Recognition`

Current research stack:

- dataset: RAVDESS speech subset
- modalities used by the project:
  - video-only files: modality `02`
  - audio-only files: modality `03`
  - vocal channel: speech (`01`)
- task: 8-class categorical emotion classification
- audio encoder: `microsoft/wavlm-base`
- video encoder: ImageNet-pretrained ResNet18
- existing fusion methods:
  - gated fusion
  - bidirectional cross-attention
  - concat / late fusion are also available in code

Main experiment-related files currently include:

- `src/data/ravdess.py`
- `src/train.py`
- `src/eval.py`
- `src/models/fusion.py`
- `src/models/wavlm_audio.py`
- `src/models/video.py`
- `tests/`

Deployment-oriented code is not part of the revision scope.

---

## 2. Fixed six-fold actor-independent protocol

RAVDESS has 24 actors. Use these fixed groups:

- `G1 = [1, 2, 3, 4]`
- `G2 = [5, 6, 7, 8]`
- `G3 = [9, 10, 11, 12]`
- `G4 = [13, 14, 15, 16]`
- `G5 = [17, 18, 19, 20]`
- `G6 = [21, 22, 23, 24]`

Each group contains two odd actor IDs and two even actor IDs, giving a simple male/female balance under the RAVDESS actor-ID convention.

Use the following fold schedule:

| Fold | Test | Validation | Training |
|---|---|---|---|
| 1 | G1 | G2 | G3+G4+G5+G6 |
| 2 | G2 | G3 | G1+G4+G5+G6 |
| 3 | G3 | G4 | G1+G2+G5+G6 |
| 4 | G4 | G5 | G1+G2+G3+G6 |
| 5 | G5 | G6 | G1+G2+G3+G4 |
| 6 | G6 | G1 | G2+G3+G4+G5 |

Per fold:

- 16 training actors
- 4 validation actors
- 4 test actors
- no actor overlap between partitions
- each actor appears exactly once in test across the six folds
- each actor appears exactly once in validation across the six folds

Do not replace this with randomly generated folds or seed-based fold assignment.

---

## 3. Dataset preflight audit

Before any production fold training, run an explicit audit over the actual local dataset.

Expected complete paired speech subset:

- 24 actors
- 60 valid paired audio-video samples per actor
- 1440 total paired samples

Expected per-actor emotion distribution:

- neutral: 4
- calm: 8
- happy: 8
- sad: 8
- angry: 8
- fearful: 8
- disgust: 8
- surprised: 8

Expected global paired distribution:

- neutral: 96
- each of the other seven emotions: 192
- total: 1440

The audit must report, at minimum:

- discovered actor IDs
- video count by actor
- audio count by actor
- valid pair count by actor
- emotion distribution by actor
- total pair count
- duplicate pair keys if any
- missing counterpart files if detectable

For production runs, fail fast if the actual paired dataset is not the expected complete 24×60 structure unless the user explicitly decides to proceed with an incomplete dataset.

---

## 4. Leakage guards

For every fold, enforce and test:

- `train_actors ∩ val_actors = ∅`
- `train_actors ∩ test_actors = ∅`
- `val_actors ∩ test_actors = ∅`
- union of train/val/test actors is exactly actors 1..24
- test groups across all six folds cover each actor exactly once
- validation groups across all six folds cover each actor exactly once

The runner should fail loudly if any invariant is violated.

---

## 5. Fold-aware output structure

Do not allow folds to overwrite each other's checkpoints.

Recommended structure:

```text
outputs/
  speaker_independent/
    fold_01/
      split.json
      audio/
        best.pt
        metrics.json
        config.json
      video/
        best.pt
        metrics.json
        config.json
      concat/
        best.pt
        metrics.json
        config.json
      chumachenko_ia/
        best.pt
        metrics.json
        config.json
      gated/
        best.pt
        metrics.json
        config.json
      xattn/
        best.pt
        metrics.json
        config.json
    ...
    fold_06/
    aggregate.json
    aggregate.csv
```

Exact filenames may differ, but the semantics must remain clear.

Each fold must save a `split.json` containing the exact train/val/test actor IDs.

Each model run must save enough config to reproduce the result.

---

## 6. Checkpoint correctness

The current training code saves the best checkpoint according to validation macro-F1 but later evaluates the in-memory model from the final epoch.

Fix this.

Required behavior:

1. train;
2. select best checkpoint by validation macro-F1;
3. save it;
4. after training/early stopping, reload that best checkpoint;
5. evaluate the test set exactly once using the reloaded best checkpoint;
6. persist the resulting test metrics.

The final reported fold score must always correspond to the best-validation checkpoint.

---

## 7. Fold-specific unimodal warm starts

The multimodal gated and cross-attention models currently support warm-starting from audio/video checkpoints.

Under speaker-independent CV:

- old random-split `best_audio.pt` and `best_video.pt` are forbidden;
- Fold N multimodal models may only warm-start from Fold N unimodal checkpoints;
- those unimodal checkpoints must have been trained using only Fold N training actors, with Fold N validation actors used for model selection;
- Fold N test actors must never influence the unimodal or multimodal training process.

Recommended order inside each fold:

1. train audio-only
2. train video-only
3. train same-protocol simple fusion baseline (concat or late; concat preferred unless there is a strong reason otherwise)
4. train adapted Chumachenko IA baseline
5. train gated fusion
6. train cross-attention fusion

---

## 8. Metrics

Per fold, record at least:

- Accuracy
- Precision
- Recall
- Macro-F1

For the final aggregate result, report:

- mean across the six folds
- standard deviation across the six folds

Do not report only one fold as the main revision result.

Prefer a machine-readable aggregate file plus a human-readable CSV/table.

If precision and recall are not currently implemented, add them with a consistent macro-averaging policy appropriate for the 8-class task and document it.

---

## 9. Existing evaluation consistency issue

The current standalone `src/eval.py` creates the test dataset with `use_wavlm=False` even when checkpoint config can specify `use_wavlm=True`.

Fix the evaluator so input preprocessing is consistent with the loaded checkpoint/model configuration.

The evaluator must not silently feed mel-style data into a WavLM model.

---

## 10. External baseline: Chumachenko et al. adapted Intermediate Attention

Use the published method:

Kateryna Chumachenko, Alexandros Iosifidis, Moncef Gabbouj,
“Self-attention fusion for audiovisual emotion recognition with incomplete data,” ICPR 2022.

Do **not** recreate their legacy software environment.

Do **not** depend on:

- PyTorch 1.9
- torchvision 0.10
- librosa 0.8
- facenet-pytorch preprocessing
- their old EfficientFace checkpoint
- their original 5-fold splits
- their published pretrained multimodal checkpoint

Instead, adapt the **Intermediate Attention (IA), 1-head fusion mechanism** into this repository's modern pipeline.

### Controlled-baseline definition

Use:

- audio representation: current WavLM sequence/features
- video representation: current ResNet18 temporal sequence/features
- same RAVDESS 8-class samples
- same fixed six actor-independent folds
- same train/validation/test actor assignments
- same general optimization/evaluation infrastructure as the project's other multimodal models where reasonable
- one attention head for the adapted Chumachenko IA baseline

### Intended semantics of the adapted IA baseline

Preserve the conceptual mechanism from the published IA implementation:

- derive bidirectional audio-to-video and video-to-audio attention relationships from intermediate modality sequences;
- convert those attention relationships into modality-specific temporal weighting/modulation;
- apply the modulation to intermediate modality features;
- pool the resulting modality representations;
- concatenate pooled audio/video representations;
- classify into 8 emotions.

The implementation should be recognizably faithful to the published IA mechanism without importing the legacy encoders or preprocessing stack.

Name it clearly, e.g.:

- CLI/model name: `chumachenko_ia`
- class name: `ChumachenkoIntermediateAttentionFusion`

Do not present it as an exact reproduction of the original paper's full pipeline. It is an **adapted external published fusion baseline**.

---

## 11. Paper/result comparison policy

The original Chumachenko result and the new adapted result are distinct facts.

- Original paper result: reported under the authors' own protocol and stack.
- New table result: generated by this repository under the fixed six-fold WavLM/ResNet18 protocol.

Do not copy the original paper's reported accuracy into the new main result table as if it were directly comparable.

---

## 12. Runner requirements

Add a simple, explicit cross-validation runner or equivalent orchestration layer.

It should support:

- selecting one fold for smoke/debug work;
- selecting one model;
- running the full set of six folds;
- clear fold/model progress logs;
- fold-aware paths;
- aggregate metrics generation;
- safe resume/skip of complete matching runs; fail on incomplete or mismatched artifacts.

Avoid a complex workflow engine. A small Python runner or shell/Python combination is sufficient.

---

## 13. Smoke testing

The first PR should not launch the full expensive production matrix.

Provide a cheap end-to-end smoke path that verifies:

- a fold is constructed correctly;
- train/val/test actors are disjoint;
- data loaders are created;
- the selected model can run forward/backward;
- checkpoint saving/loading works;
- best-checkpoint reload is used for test;
- metrics serialization works;
- the adapted Chumachenko IA model produces 8-class logits.

A one-batch or tiny-epoch smoke mode is acceptable if clearly separated from production settings.

---

## 14. Tests

Add automated tests for at least:

1. fixed fold definitions
2. all six fold invariants
3. actor disjointness
4. actor coverage exactly once in test
5. actor coverage exactly once in validation
6. dataset audit behavior on complete and incomplete synthetic data
7. fold-specific output paths
8. best-checkpoint reload path
9. WavLM evaluator consistency
10. Chumachenko IA tensor shapes and forward pass
11. aggregate mean/std calculation

Tests should not require the full RAVDESS dataset.

---

## 15. Out of scope for this first PR

Do not implement unless directly needed:

- new audio/video encoders
- new attention architectures beyond the adapted published baseline
- CLIP-style alignment experiments
- emotion-prior bias experiments
- gate-interpretability analysis
- noise ablations
- face-crop ablations
- major hyperparameter search
- backend changes
- frontend changes
- Docker changes
- ONNX/export changes
- paper prose editing
- plagiarism reduction
- IEEE template formatting


## 16. First-PR implementation conventions

- Entry points: `revision.audit` and `revision.run` with `PYTHONPATH=src`.
- Precision, recall, and F1 are macro averages over labels 0..7, including absent
  classes, with undefined terms set to zero. Best-checkpoint selection uses this
  same macro-F1. Accuracy is the fraction of correct samples.
- Aggregate standard deviation is sample standard deviation (`ddof=1`). A single
  fold has `std: null`; incomplete fold coverage is explicitly flagged and must
  not be used as the main revision result.
- `split.json` records the exact actor split under `split`, plus protocol, source
  fingerprint, Git commit, library versions, dataset manifest fingerprint, mode,
  common settings and the full resolved per-model profile catalog. Model configs
  save their profile ID, resolved profile and all trainer arguments; checkpoint
  records also contain selected epoch and validation F1.
- The data audit checks the exact designed speech pair keys, file counts, and
  filename validity. Its manifest fingerprint covers relative paths, file sizes,
  and modification times. It does not certify media decoding or hash file contents.
- Fusion selection automatically trains missing audio/video prerequisites, then
  loads only their current-fold checkpoints with matching provenance and strict
  state-dict compatibility. Existing completed prerequisites can be reused only
  under the identical experiment identity. Every complete matching selected run
  is skipped on resume. All existing run directories are checked before training;
  incomplete/changed runs fail loudly and require a fresh output root.
- Smoke mode uses 24 synthetic decodable pairs (one per actor), a tiny randomly
  initialized WavLM test fixture, and ResNet18 without pretrained weights. It
  runs two epochs with two frames and no augmentation/face crop. Fusion stage 1
  lasts one epoch, exercising stage 2 in epoch 2; workers remain 0. A root mode
  sentinel and provenance checks prevent mixing it with production evidence.
- Production WavLM initialization fails if pretrained weights are unavailable;
  silent random-weight fallback is forbidden. This also makes missing pretrained
  weights visible in the shared trainer.
- Adapted IA projects encoder sequences into 128 dimensions, applies independent
  one-head query/key attention in each direction, sums normalized attention over
  queries, multiplies the opposite stream by these temporal weights, mean-pools,
  concatenates, and uses an eight-class linear classifier. The authors' temporal
  convolution stages and modality-dropout experiments are not reproduced. The
  unnormalized query sum is intentional and preserves the published IA weighting
  convention across unequal audio/video sequence lengths.

See `REVISION_RUNS.md` for commands and validation evidence.

Standalone evaluation initializes encoder structure without downloading pretrained
weights, then strictly loads the complete saved model state. This supports offline
checkpoint evaluation while production training still requires pretrained weights.


## 17. Canonical production profiles and strict resume

The user-approved profiles supersede the first infrastructure commit's generic
single-stage configuration. `revision.profiles` resolves the explicit audio,
video, gated, xattn, adapted IA and concat profiles. Exact default resolved
configs/IDs are recorded in `REVISION_PROFILES.json`; optimization/architecture
parameters are fixed by profile rather than generic runner CLI overrides.
Production defaults to loader workers -1 (the existing auto policy).

- Audio: 20 epochs, batch 16, LR 1e-3, decay 1e-4, WavLM stage 2,
  backbone LR 3e-5, cosine scheduling, patience 10.
- Video: 20 epochs, batch 16, LR 1e-3, decay 1e-4, cosine, patience 10.
- Gated, concat and IA: 30 epochs, batch 8, LR 3e-4, decay 1e-4, two-stage,
  stage 1 for 5 epochs, audio/video backbone LR 1e-5, 2 unfrozen WavLM layers,
  1 unfrozen video block, cosine, patience 8.
- XAttn: 35 epochs, batch 8, LR 2e-4, decay 2e-4, gated head, d_model 96,
  4 heads, attention dropout/drop path 0.1, smoothing 0.05, two-stage,
  stage 1 for 6 epochs, audio/video backbone LR 8e-6, 2 WavLM layers,
  1 video block, cosine, patience 10.

The evaluation split changes; the paper's Gated/XAttn architecture and training
definitions are preserved. IA's one-head query-summed weighting is unchanged:
no normalization and no original softhard modality-dropout pipeline.

Each model's full resolved config and versioned profile ID appear in config JSON,
checkpoint and metric provenance. Every fold also stores the entire profile
catalog so the same model profile can be checked across folds while different
models retain their intended distinct settings. Resume/warm starts/aggregation
use the same full expected-config and completed-artifact checks.

Complete matching runs are skipped; absent runs are trained. Existing incomplete,
corrupt or mismatched runs anywhere in the output root fail before new training.
Completion requires readable best checkpoint, metrics, config and fold split,
matching full config/provenance, consistent best epoch/validation score/path, and
finite bounded test metrics. Partial epoch recovery is not implemented. Initial
single-stage harness artifacts cannot be reused under these new profiles.
