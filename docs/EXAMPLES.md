# EXAMPLES.md

## 1. GOOD fold definition

```json
{
  "fold": 1,
  "train_actors": [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
  "val_actors": [5,6,7,8],
  "test_actors": [1,2,3,4]
}
```

Why this is good:

- 16/4/4 actors
- all three partitions are disjoint
- fixed and reproducible
- matches the project decision exactly

## BAD fold definition

```python
random.shuffle(actor_ids)
train = actor_ids[:16]
val = actor_ids[16:20]
test = actor_ids[20:24]
```

Why this is bad:

- the project explicitly chose fixed groups;
- the paper and response need deterministic, easily described folds.

---

## 2. GOOD checkpoint usage

```text
fold_03/audio/best.pt
        ↓
fold_03/gated warm start
        ↓
train on fold_03 train actors only
        ↓
select by fold_03 validation F1
        ↓
reload fold_03/gated/best.pt
        ↓
evaluate once on fold_03 test actors
```

## BAD checkpoint usage

```text
old outputs/best_audio.pt from random split
        ↓
fold_03 gated fusion
```

Why this is bad:

The old audio model may already have seen Fold 3 test actors.

---

## 3. GOOD external baseline framing

> Chumachenko et al. Intermediate Attention fusion was reimplemented under the same WavLM/ResNet18 encoders and the same six actor-independent folds used for all methods.

This is accurate.

## BAD external baseline framing

> We reproduced Chumachenko et al. and obtained X%.

This is misleading because the project is not reproducing their full MFCC/EfficientFace pipeline, original preprocessing, or original five-fold protocol.

---

## 4. GOOD dataset preflight output

```text
RAVDESS paired-data audit
Actor 01: video=60 audio=60 paired=60
Actor 02: video=60 audio=60 paired=60
...
Actor 24: video=60 audio=60 paired=60

Total paired: 1440

Emotion distribution:
neutral      96
calm        192
happy       192
sad         192
angry       192
fearful     192
disgust     192
surprised   192

PASS actor completeness
PASS pair-key uniqueness
PASS expected class counts
```

## BAD dataset behavior

```text
Actor 17 paired=58
continuing training...
```

Production training should not silently proceed.

---

## 5. GOOD aggregate result

```text
method,accuracy_mean,accuracy_std,precision_mean,precision_std,recall_mean,recall_std,macro_f1_mean,macro_f1_std
gated,0.781,0.034,0.776,0.037,0.779,0.035,0.773,0.039
```

and preserve raw fold results separately.

## BAD aggregate result

```text
gated accuracy = 0.824
```

if 0.824 is merely the best fold.

---

## 6. GOOD test semantics

```python
best_path = ...
train(...)
load_best_checkpoint(best_path)
test_metrics = evaluate(test_loader)
```

## BAD test semantics

```python
train(...)
test_metrics = evaluate(current_model)
```

when `current_model` may be a later, worse validation epoch than the saved best checkpoint.

---

## 7. GOOD adapted Chumachenko IA behavior

Conceptually:

```text
WavLM sequence ──────┐
                     ├─ bidirectional intermediate attention
ResNet18 sequence ───┘
          ↓
temporal importance/modulation of each modality
          ↓
pool audio + pool video
          ↓
concatenate
          ↓
8-class classifier
```

The mechanism should preserve the published IA idea while using the project's shared encoders.

## BAD adaptation

```text
torch.cat([audio_embedding, video_embedding]) -> MLP
```

and labeling it “Chumachenko IA”.

That would collapse the baseline into a simple concat model and lose the defining published mechanism.

---

## 8. GOOD scope behavior

A PR that changes:

- fold configuration
- experiment runner
- metrics
- evaluation correctness
- adapted baseline
- tests
- docs

is in scope.

## BAD scope behavior

A PR that additionally:

- redesigns the frontend
- changes Docker
- introduces a new transformer backbone
- modifies the inference server
- adds CLIP alignment experiments

is scope creep.
