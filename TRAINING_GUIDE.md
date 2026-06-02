# Training Guide

This guide documents the exact training configurations used for the paper experiments, along with environment setup, data preparation, and evaluation instructions.

## Environment Setup

```bash
# Python 3.10+ required
uv sync          # recommended
# or
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchaudio`, `torchvision`, `transformers` (WavLM), `opencv-python`, `mediapipe`, `wandb`.

## Data Preparation

Download [RAVDESS](https://zenodo.org/record/1188976) and extract under `data/`:

```
data/
  Actor_01/
    02-01-01-01-01-01-01.mp4    # face-only video (modality 02)
    ...
  Actor_02/
  ...
  Actor_24/
  Noise/
    noise.wav                   # optional bar background noise for augmentation
```

The data pipeline automatically pairs video files (modality 02) with matching audio files (modality 03) by matching on `(vocal_channel, emotion, intensity, statement, repetition, actor)`.

## Paper Experiments (Exact Commands)

All runs use stratified splitting and W&B logging. Remove `--wandb` if not using W&B.

### Step 1 — Audio-only baseline (WavLM)

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion audio \
  --use_wavlm \
  --split_mode stratified \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 10 \
  --use_face_crop \
  --wandb
```

Saves checkpoint to `outputs/best_audio.pt`.

### Step 2 — Video-only baseline (ResNet18)

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion video \
  --split_mode stratified \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 10 \
  --use_face_crop \
  --wandb
```

Saves checkpoint to `outputs/best_video.pt`.

### Step 3 — Gated fusion (best result, 93.33% accuracy)

Uses warm-start from pre-trained unimodal checkpoints and two-stage fine-tuning.

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion gated \
  --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training \
  --stage1_epochs 5 \
  --lr 3e-4 \
  --audio_backbone_lr 1e-5 \
  --video_backbone_lr 1e-5 \
  --fusion_unfreeze_wavlm_layers 2 \
  --fusion_unfreeze_video_blocks 1 \
  --split_mode stratified \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --epochs 30 \
  --batch_size 8 \
  --weight_decay 1e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 8 \
  --use_face_crop \
  --wandb
```

### Step 4 — Cross-attention fusion (92.00% accuracy)

Uses `xattn_head=gated`, `d_model=96`, label smoothing, and longer warm-start.

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion xattn \
  --xattn_head gated \
  --xattn_d_model 96 \
  --xattn_heads 4 \
  --xattn_attn_dropout 0.1 \
  --xattn_stochastic_depth 0.1 \
  --label_smoothing 0.05 \
  --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training \
  --stage1_epochs 6 \
  --lr 2e-4 \
  --audio_backbone_lr 8e-6 \
  --video_backbone_lr 8e-6 \
  --fusion_unfreeze_wavlm_layers 2 \
  --fusion_unfreeze_video_blocks 1 \
  --split_mode stratified \
  --train_ratio 0.75 \
  --val_ratio 0.15 \
  --epochs 35 \
  --batch_size 8 \
  --weight_decay 2e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 10 \
  --use_face_crop \
  --wandb
```

## Two-Stage Training Strategy

Both multimodal models use a two-stage fine-tuning strategy:

- **Stage 1** (fusion head warm-up): Both encoders are frozen. Only the fusion head is trained. This prevents early gradient interference between the unimodal backbone weights.
- **Stage 2** (selective unfreeze): The last 2 WavLM transformer layers and the last ResNet18 block are unfrozen with a much lower learning rate (`8e-6` or `1e-5`), while the fusion head continues at the higher rate.

## Audio Preprocessing

- Waveform loaded at 16 kHz, padded or trimmed to 3 seconds → shape `[1, 48000]`
- Curriculum noise augmentation during training: 50% clean, 40% medium noise (20/15/10 dB SNR), 10% heavy noise (5 dB SNR)
- Real bar noise from `data/Noise/noise.wav`; falls back to Gaussian noise if unavailable

## Video Preprocessing

- 8 frames uniformly sampled per clip, resized to 112×112, normalized with ImageNet statistics
- Face detection via MediaPipe on the first frame; bounding box reused for remaining frames
- Visual augmentation during training: Gaussian blur (kernel 3/5/7), brightness scaling [0.2, 0.6], light additive Gaussian noise

## Evaluation

```bash
# Evaluate a saved checkpoint
uv run python src/eval.py \
  --checkpoint outputs/best_gated.pt \
  --data_root data \
  --num_classes 8 \
  --split_mode stratified
```

## Reproduce Table 1 (parameters, GFLOPs, inference time)

```bash
uv run python src/benchmark_table6.py
```

This script loads the four model architectures, applies the same freeze policy as training, and measures parameters, GFLOPs (via `torchinfo`), and wall-clock inference time over 100 forward passes.

## Key Hyperparameters Summary

| Hyperparameter | Audio-only | Video-only | Gated fusion | Cross-attention |
|---|---|---|---|---|
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 | 3e-4 (head), 1e-5 (backbone) | 2e-4 (head), 8e-6 (backbone) |
| Weight decay | 1e-4 | 1e-4 | 1e-4 | 2e-4 |
| Batch size | 16 | 16 | 8 | 8 |
| Max epochs | 20 | 20 | 30 | 35 |
| Early stopping patience | 10 | 10 | 8 | 10 |
| Label smoothing | — | — | — | 0.05 |
| d_model (cross-attention) | — | — | — | 96 |
| Attention heads | — | — | — | 4 |
| Attention dropout | — | — | — | 0.1 |
| Stochastic depth | — | — | — | 0.1 |
| Scheduler | Cosine annealing | Cosine annealing | Cosine annealing | Cosine annealing |
| Train/Val/Test split | 70/15/15 | 70/15/15 | 70/15/15 | 75/15/10 |

## Inference Deployment

### Direct Python

```bash
EMO_CHECKPOINT=outputs/best_gated.pt python src/inference_server.py
# POST http://localhost:8000/predict  (multipart/form-data: file=<video.mp4>)
```

### Redis-queue batch worker (for higher throughput)

```bash
redis-server &

EMO_REDIS_URL=redis://localhost:6379/0 \
EMO_CHECKPOINT=outputs/best_gated.pt \
EMO_BATCH_SIZE=8 \
EMO_BATCH_TIMEOUT_MS=20 \
python src/inference_worker.py &

EMO_REDIS_URL=redis://localhost:6379/0 \
python src/inference_server.py
```

### Docker

```bash
docker-compose up
# Web UI available at http://localhost:80
```

### ONNX export (optional, for faster CPU inference)

```bash
# Export to ONNX
python src/export_optimized_model.py \
  --checkpoint outputs/best_gated.pt \
  --output outputs/best_gated.onnx

# Export with INT8 quantization
python src/export_optimized_model.py \
  --checkpoint outputs/best_gated.pt \
  --output outputs/best_gated_int8.onnx \
  --quantize_int8

# Run with ONNX backend
EMO_INFERENCE_BACKEND=onnx \
EMO_ONNX_MODEL_PATH=outputs/best_gated_int8.onnx \
python src/inference_worker.py
```
