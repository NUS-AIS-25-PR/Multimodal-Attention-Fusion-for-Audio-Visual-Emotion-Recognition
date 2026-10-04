# DECISIONS.md

## D1 — Prioritize two revision tasks only

**Decision**

The one-week engineering effort prioritizes:

1. speaker/actor-independent re-evaluation;
2. one practical external published baseline.

Other reviewer requests are secondary and may be implemented only if time remains.

**Reason**

The reviewers' central methodological concern is actor leakage, and they explicitly asked for a stronger external comparison. These two items provide the largest revision value per unit time.

**Rejected alternatives**

- trying to address every reviewer comment in code immediately;
- broad architecture exploration;
- extensive ablation work before the core evaluation is corrected.

---

## D2 — Use fixed six-fold actor-independent CV

**Decision**

Use six fixed groups of four actors:

- G1 = 01–04
- G2 = 05–08
- G3 = 09–12
- G4 = 13–16
- G5 = 17–20
- G6 = 21–24

Fold N uses `GN` for test and the next group cyclically for validation.

**Reason**

RAVDESS speech actors have equal designed sample counts and equal class composition, so no sample-count balancing algorithm is required. The fixed groups are simple, reproducible, easy to describe, and keep each four-actor group gender-balanced under the odd/even actor-ID convention.

**Rejected alternatives**

- random actor grouping with a seed;
- one single 18/3/3 actor split;
- random sample-level stratification;
- LOAO/24-fold evaluation, because the one-week time budget makes six folds a better cost/rigour tradeoff.

---

## D3 — Require complete-data preflight before production runs

**Decision**

Expect 60 valid pairs per actor and 1440 total paired speech samples. Production CV should fail if the dataset is incomplete unless the user explicitly overrides the decision.

**Reason**

The project's pairing logic can silently drop unmatched files. A formal audit prevents incomplete local data from creating misleading fold sizes or class distributions.

**Rejected alternatives**

- silently continuing with whatever files happen to pair;
- relying only on official RAVDESS counts without checking the local copy.

---

## D4 — Do not reuse old random-split checkpoints

**Decision**

All fold-specific unimodal and multimodal checkpoints must be trained inside the same actor-independent fold.

**Reason**

An old unimodal checkpoint trained with random sample splitting may already contain information from actors later designated as test actors. Warm-starting from it would invalidate the speaker-independent claim.

**Rejected alternatives**

- using the old `best_audio.pt` / `best_video.pt` for speed;
- freezing an old leaked encoder and training only a new fusion head.

---

## D5 — Report six-fold mean ± standard deviation

**Decision**

The main revised results will use six-fold aggregate metrics rather than a single test split.

**Reason**

The reviewers explicitly requested fold variability, and RAVDESS is small enough that split-specific performance can vary materially.

**Rejected alternatives**

- one-fold accuracy only;
- selecting the best fold;
- reporting only the fold used by another paper.

---

## D6 — Adapt Chumachenko et al. ICPR 2022 Intermediate Attention as the external baseline

**Decision**

Use the published Chumachenko et al. Intermediate Attention (IA), one-head mechanism as the external baseline, adapted to the project's current WavLM + ResNet18 pipeline and fixed six-fold actor-independent protocol.

**Reason**

It is a peer-reviewed audio-visual RAVDESS fusion method with public code, its task is closely aligned with this paper, and adapting the fusion mechanism allows a controlled comparison while avoiding obsolete dependency problems.

**Rejected alternatives**

- full legacy reproduction of the original Chumachenko software stack;
- Chumachenko pretrained checkpoints;
- reproducing their original five folds;
- Praveen et al. JCA as the main baseline, because it targets a different dataset/task (dimensional emotion regression);
- Ibrahim et al. 2026 as the primary implementation target, because its full pipeline is much more complex and its reproducibility package is not a straightforward end-to-end training implementation;
- an unreviewed GitHub-only project as the primary academic external baseline.

---

## D7 — Do not recreate the legacy PyTorch environment

**Decision**

Do not install PyTorch 1.9 / torchvision 0.10 as the baseline environment.

**Reason**

The development GPU is an RTX 5080-class Blackwell device requiring a modern CUDA/PyTorch stack. The Chumachenko fusion code itself uses ordinary PyTorch operations and can be migrated cleanly.

**Rejected alternatives**

- separate old CUDA training environment;
- CPU training under the old stack;
- maintaining two incompatible environments only to preserve legacy preprocessing.

---

## D8 — Keep shared encoders for the adapted external baseline

**Decision**

The adapted Chumachenko IA baseline uses the same WavLM audio encoder and ResNet18 video encoder as the current paper models.

**Reason**

This turns the new table into a controlled fusion comparison. Performance differences are less confounded by backbone strength.

**Rejected alternatives**

- EfficientFace + MFCC for only the external baseline;
- swapping all paper models to Chumachenko's original encoders.

---

## D9 — First PR builds infrastructure; production training happens after review

**Decision**

Codex should implement, test, and push the experiment harness and adapted baseline first. Do not immediately launch the full six-fold production matrix.

**Reason**

A leakage, checkpoint, or split bug discovered after a 10–20 hour run would waste the limited revision window.

**Rejected alternatives**

- editing the code and starting all training in the same step;
- accepting an unreviewed fold runner because the current `actor` split already exists.

---

## D10 — Fix best-checkpoint test semantics

**Decision**

Final test evaluation must reload the checkpoint selected by best validation macro-F1.

**Reason**

The current training loop saves the best checkpoint but evaluates the final in-memory model. Those are not guaranteed to be the same model.

**Rejected alternatives**

- keep testing the final epoch;
- use the lowest training-loss model;
- choose a checkpoint based on test performance.

---

## D11 — Freeze non-revision systems

**Decision**

Backend, frontend, Docker, inference serving, ONNX, and deployment features are out of scope.

**Reason**

They do not address reviewer concerns and create unnecessary regression risk.

**Rejected alternatives**

- opportunistic cleanup/refactor while touching the repository.


---

## D12 — Preserve canonical per-model production definitions

**Decision**

Use the explicit user-approved profiles in `SPEC.md` section 17 and
`REVISION_PROFILES.json`. Audio fine-tunes WavLM at stage 2; Gated/XAttn retain
canonical two-stage optimization and XAttn's gated 96-dimensional head. Adapted
IA and concat share Gated optimization. IA remains one-head/query-summed without
normalization or original softhard modality dropout. Production loader workers
use the existing auto policy by default.

**Reason and superseded behavior**

The first infrastructure PR used generic single-stage defaults. That was adequate
for an infrastructure smoke but would silently change the paper's production
training definitions. This explicit decision supersedes those defaults; the
revision must change the evaluation protocol while preserving model definitions.

---

## D13 — Resume only completed runs with exact provenance

**Decision**

Skip complete matching runs, including explicitly selected models, and train
only absent runs. Fail before training if any existing run is incomplete,
corrupt, or mismatches full config, profile/catalog, fold, source or dataset.
Persist profile IDs and full resolved configs in every artifact, and recheck
across folds during aggregation. Do not recover partial epochs or mix prior
single-stage artifacts with the new profiles.

**Reason and superseded behavior**

The first infrastructure version rejected all existing selected runs and only
reused completed unimodal dependencies. Long CV runs require safe completed-fold
resume without redoing training or changing scientific provenance.
