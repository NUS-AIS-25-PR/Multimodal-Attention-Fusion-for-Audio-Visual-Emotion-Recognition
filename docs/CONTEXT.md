# CONTEXT.md

## Conference situation

The paper:

**Multimodal Attention Fusion for Audio-Visual Emotion Recognition**

has been accepted to IEEE SPMB 2026 with **mandatory revision**.

The revised PDF and a summary of key changes are due October 15, 2026.

The engineering work in this handoff is focused on the paper's experimental revision only. The professor will handle plagiarism reduction and much of the prose/formatting work.

## Reviewer concerns most relevant to code

Across the reviews, the dominant concerns were:

- the novelty claims are too strong for standard gated/cross-attention mechanisms;
- random sample-level splitting can put the same actor in both train and test;
- the model may therefore exploit actor/speaker identity rather than emotion;
- subject-independent validation should be the main result;
- per-fold variability should be reported;
- at least one strong external baseline should be included.

Reviewer 3 specifically stated that actor-independent results plus a strong external baseline would make the claims considerably more persuasive.

## Existing paper setup

The current paper uses:

- RAVDESS
- 24 actors
- 8 classes
- modality 02 video-only paired with modality 03 audio-only
- WavLM-base audio encoder
- ResNet18 video encoder
- gated fusion
- bidirectional cross-attention fusion

The paper reports approximately:

- audio-only accuracy: 0.8133
- video-only accuracy: 0.7778
- cross-attention accuracy: 0.9200
- gated fusion accuracy: 0.9333

These are old random-split results and are not the target scores for the revision.

## Important discrepancy discovered in the current reproducibility material

The repository's recorded paper commands do not use identical random split ratios for all methods:

- audio/video/gated: 70/15/15
- cross-attention: 75/15/10

The revision should replace this old comparison with the unified actor-independent protocol rather than trying to preserve or patch the old table.

## Existing actor split support

The repository already has actor-based splitting in `src/data/ravdess.py` and CLI actor lists in `src/train.py`.

This handoff is therefore not asking for a new dataset architecture. It is asking for:

- fixed fold definitions;
- orchestration;
- validation guards;
- result aggregation;
- correctness fixes.

## Existing training correctness issue

The current training loop saves the best validation checkpoint but performs final test evaluation on the in-memory model after the last/early-stopping epoch.

This must be corrected before production experiments.

## Existing standalone evaluation issue

`src/eval.py` currently constructs the dataset with `use_wavlm=False` while the model may be reconstructed as WavLM from checkpoint config.

This mismatch must be corrected.

## External baseline research

### Chumachenko et al., ICPR 2022

Chosen implementation target:

**Intermediate Attention (IA), one head, adapted to the current shared encoders.**

Why chosen:

- peer-reviewed
- public code
- audio-visual emotion recognition
- RAVDESS support
- 8-class setting
- speaker-independent framing
- fusion mechanism is close enough to be a meaningful comparison

Why the original environment is not used:

Official repo pins an old stack around:

- Python 3.9
- PyTorch 1.9
- torchvision 0.10
- librosa 0.8.1

The current development machine uses a modern RTX 5080-class GPU. Recreating the old CUDA/PyTorch stack is not a good use of the one-week revision window.

The official preprocessing also depends on facenet-pytorch/MTCNN and EfficientFace checkpoints, adding avoidable environment and availability risk.

The fusion logic itself is ordinary PyTorch and can be implemented in the modern project stack.

### Important wording

The new baseline is not an exact reproduction of the original full Chumachenko pipeline.

Use wording such as:

- adapted external baseline
- reimplemented Intermediate Attention mechanism
- evaluated under a common WavLM/ResNet18 backbone and common actor-independent folds

Do not call it an exact reproduction.

### Ibrahim et al., Scientific Reports 2026

This work is particularly relevant because Reviewer 1 explicitly cited it and it demonstrates a large gap between random-split and strict speaker-independent evaluation.

It should be used in the paper's related-work/discussion framing, but it is not the primary implementation target for the one-week coding work because its complete pipeline is substantially more complex.

### Praveen et al., CVPR Workshops 2022

Not chosen as the implementation baseline because it targets dimensional emotion recognition on a different dataset/protocol.

## Time constraint

Only about one week is available for the revision.

This is why the project optimizes for:

- methodological correctness;
- reproducibility;
- low implementation risk;
- fast review of code before GPU time is spent.

## Recommended implementation sequence

1. create branch
2. add fold specification + tests
3. add data audit + tests
4. fix checkpoint/evaluator correctness
5. add fold-aware result paths/metadata
6. add metrics aggregation
7. add adapted Chumachenko IA baseline
8. run unit tests
9. run one cheap smoke fold/model
10. push PR
11. review PR before starting full training
