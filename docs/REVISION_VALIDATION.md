# First-PR validation evidence

Validation was performed on the feature source before delivery, using the
existing Python 3.10 environment and PyTorch 2.10.0+cu130 on an RTX 5080 Laptop
GPU. No legacy environment, random-split warm start, or production matrix was used.

| Check | Result |
|---|---|
| Python compile check | Pass |
| Relevant revision/data/attention tests | 20 passed; no skips |
| Full repository discovery | 27 passed, 1 pre-existing backend failure; no skips |
| Backend tests on untouched `d50f07c` | 7 passed, same 1 failure |
| Fold 1 synthetic smoke: audio, video, adapted IA | Pass; one epoch per model |
| Offline standalone IA evaluation | Pass; identical saved test metrics |
| Smoke artifact/provenance assertions | Pass |
| Local `data/` completeness audit | Expected refusal: 0/1440 pairs |

Commands executed (full logs remain in ignored `tmp/`, artifacts in `outputs/`):

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q \
  src/train.py src/eval.py src/revision src/models/chumachenko_ia.py \
  src/models/wavlm_audio.py tests/test_revision.py
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_revision.py'
PYTHONPATH=src:tests OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  test_revision test_data_services test_attention_integration
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src .venv/bin/python -m revision.audit \
  --data-root data --output outputs/local_ravdess_audit.json
PYTHONPATH=src .venv/bin/python -m revision.run \
  --smoke --fold 1 --model chumachenko_ia \
  --output-root outputs/revision_smoke_reviewed
PYTHONPATH=src OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python src/eval.py \
  --data_root outputs/revision_smoke_reviewed/synthetic_media \
  --checkpoint outputs/revision_smoke_reviewed/fold_01/chumachenko_ia/best.pt \
  --num_workers 0
.venv/bin/python /tmp/check_revision_artifacts.py
git diff --check -- README.md src
git diff --cached --check
```

The temporary artifact assertion script compared saved and standalone metrics,
checked 16/4/4 actor disjointness/coverage, best epoch/checkpoint paths, fold-local
warm-start paths, matching config/split metadata, and incomplete aggregate flags
with null single-fold standard deviations.

To establish that the unrelated backend failure predates this PR:

```bash
git worktree add --detach /tmp/mer-revision-base-validation d50f07c
OMP_NUM_THREADS=1 /home/louis/projects/MultimodalEmotionRecognition/.venv/bin/python \
  -m unittest discover -s /tmp/mer-revision-base-validation/tests \
  -p test_backend_services.py
git worktree remove /tmp/mer-revision-base-validation
```

The failing test is
`TestStreamingEmotionSession.test_session_builds_sliding_window_and_updates_cadence`
(line 176): expected 2 buffered frames, actual 4 on both versions. Backend source
and tests remain unchanged.

The cheap smoke has 16 train / 4 validation / 4 test synthetic pairs. Train actors
09–24, validation actors 05–08 and test actors 01–04 are disjoint. IA loads only
`fold_01/audio/best.pt` and `fold_01/video/best.pt`, with strict matching provenance.
No old random-split checkpoint is reused. Each model reloads its best validation
checkpoint at epoch 1 before test scoring. IA has eight logits; saved and offline
standalone test metrics both equal:

```json
{"accuracy": 0.25, "precision": 0.03125, "recall": 0.125, "macro_f1": 0.05}
```

These synthetic scores are not research results. The local complete RAVDESS data,
production pretrained initialization, full-duration GPU memory/throughput, and
final six-fold training have not been validated. Review the adapted IA weighting
convention and shared optimization policy before production (see `REVISION_RUNS.md`).
