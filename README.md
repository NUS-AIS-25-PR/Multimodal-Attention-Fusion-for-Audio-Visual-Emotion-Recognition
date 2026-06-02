# Multimodal Attention Fusion for Audio-Visual Emotion Recognition

Implementation of the paper *"Multimodal Attention Fusion for Audio-Visual Emotion Recognition"* (IEEE SPMB 2026).

This repository provides the full training pipeline, evaluation scripts, inference server, and pre-trained model benchmarks for audio-visual emotion recognition on the RAVDESS dataset.

## Results

All results are on the RAVDESS 8-class test set using a stratified split (75/15/10 train/val/test for cross-attention; 70/15/15 for other variants). Inference time is measured at batch size 1 on an NVIDIA RTX 5080 Laptop GPU.

| Method | Accuracy | Precision | Recall | F1 | Params (M) | GFLOPs | Inference (ms) |
|---|---|---|---|---|---|---|---|
| Audio only (WavLM) | 0.8133 | 0.8117 | 0.8167 | 0.8097 | 94.98 | 7.42 | 10.4 |
| Video only (ResNet18) | 0.7778 | 0.7913 | 0.7792 | 0.7699 | 11.18 | 3.88 | 1.2 |
| **Gated fusion** | **0.9333** | **0.9343** | **0.9375** | **0.9321** | 106.62 | 11.30 | 11.9 |
| Cross-attention fusion | 0.9200 | 0.9275 | 0.9250 | 0.9200 | 106.40 | 11.30 | 12.0 |

## Architecture

The framework consists of:

- **Audio encoder**: Pretrained `microsoft/wavlm-base` with temporal mean pooling, producing a 768-dimensional clip-level embedding or a token sequence for cross-attention.
- **Video encoder**: ResNet18 (ImageNet pretrained, final FC removed) applied per frame over 8 uniformly sampled frames, producing a 512-dimensional embedding or 8-token sequence.
- **Gated fusion**: Projects both modalities to a 256-dimensional shared space, then computes a learned scalar gate to weight audio vs. video contributions.
- **Cross-attention fusion**: Projects both modalities to a 96-dimensional attention space, applies bidirectional multi-head attention (4 heads) between audio and video token sequences, then concatenates the gated-pooled outputs for classification.

```
Input audio ──► WavLM encoder ──► [768×1 or T×768]
                                         │
                                  ┌──────┴──────┐
                                  │  Fusion     │──► 8-class logits
                                  └──────┬──────┘
Input video ──► ResNet18 encoder ──► [512×1 or T×512]
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for exact training commands and hyperparameters.

## Dataset

[RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song).

- 24 professional actors, 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- This project uses face-only video files (modality 02) paired with voice-only audio files (modality 03)
- Audio sampled at 16 kHz, video at 30 fps (1920×1080), 8 frames uniformly sampled per clip

Download and place the dataset under `data/`:

```
data/
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

## Setup

```bash
# Clone and install dependencies (requires Python 3.10+)
git clone https://github.com/Wionerlol/MultimodalEmotionRecognition.git
cd MultimodalEmotionRecognition

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Quick Start

### Reproduce paper results

```bash
# 1. Train audio-only baseline (WavLM)
uv run python src/train.py \
  --data_root data --num_classes 8 --fusion audio \
  --use_wavlm --split_mode stratified \
  --epochs 20 --batch_size 16 --wandb

# 2. Train video-only baseline (ResNet18)
uv run python src/train.py \
  --data_root data --num_classes 8 --fusion video \
  --split_mode stratified \
  --epochs 20 --batch_size 16 --wandb

# 3. Train gated fusion (best result)
uv run python src/train.py \
  --data_root data --num_classes 8 --fusion gated \
  --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training --stage1_epochs 5 \
  --lr 3e-4 --audio_backbone_lr 1e-5 --video_backbone_lr 1e-5 \
  --fusion_unfreeze_wavlm_layers 2 --fusion_unfreeze_video_blocks 1 \
  --split_mode stratified --epochs 30 --batch_size 8 \
  --weight_decay 1e-4 --use_cosine_annealing \
  --early_stopping_patience 8 --wandb

# 4. Train cross-attention fusion
uv run python src/train.py \
  --data_root data --num_classes 8 --fusion xattn \
  --xattn_head gated --xattn_d_model 96 --xattn_heads 4 \
  --xattn_attn_dropout 0.1 --xattn_stochastic_depth 0.1 \
  --label_smoothing 0.05 --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training --stage1_epochs 6 \
  --lr 2e-4 --audio_backbone_lr 8e-6 --video_backbone_lr 8e-6 \
  --fusion_unfreeze_wavlm_layers 2 --fusion_unfreeze_video_blocks 1 \
  --split_mode stratified --train_ratio 0.75 --val_ratio 0.15 \
  --epochs 35 --batch_size 8 --weight_decay 2e-4 \
  --use_cosine_annealing --early_stopping_patience 10 --wandb
```

### Benchmark (reproduce Table 1)

```bash
uv run python src/benchmark_table6.py
```

### Inference server

```bash
# Start the FastAPI inference server
EMO_CHECKPOINT=outputs/best_gated.pt python src/inference_server.py

# Or with Docker
docker-compose up
```

## Project Structure

```
src/
  models/
    wavlm_audio.py     # WavLM audio encoder (used in paper)
    video.py           # ResNet18 video encoder
    fusion.py          # Gated fusion and cross-attention fusion
    temporal.py        # Temporal pooling modules
    audio.py           # Mel-spectrogram encoder (legacy, not used in paper)
  data/
    ravdess.py         # Dataset loading, pairing, and splitting
  train.py             # Training pipeline (EmotionTrainer)
  eval.py              # Standalone evaluation
  benchmark_table6.py  # Reproduces Table 1 metrics
  inference_server.py  # FastAPI inference server
  inference_worker.py  # Redis-queue batch inference worker
  export_optimized_model.py  # ONNX export and INT8 quantization

backend/               # FastAPI app for Docker deployment
frontend/              # Web UI for real-time prediction
assets/
  wandb_chart/         # Training curves and confusion matrices
  wandb_log/           # W&B run configs and logs (for reproducibility)
  diagrams/            # Architecture diagrams
  examples/            # Processed audio/video samples
```

## Experimental Components (not in paper)

The codebase contains several modules that were explored during development but are not part of the reported experiments:

| Module | Location | Description |
|---|---|---|
| `AudioNet`, `AudioResNet18`, `AudioCNN` | `src/models/audio.py` | Mel-spectrogram audio encoder. Replaced by WavLM in final experiments. |
| `ClipStyleAlignment` | `src/models/fusion.py` | CLIP-style contrastive pre-fusion alignment. Conflicted with classification loss; abandoned. |
| `EmotionPriorBiasAdapter` | `src/models/fusion.py` | Emotion-prior-conditioned cross-attention bias. Showed no consistent improvement on RAVDESS. |
| Actor-based split | `src/data/ravdess.py` | Speaker-independent evaluation split. Implemented but results not reported in paper. |

## Citation

```bibtex
@inproceedings{liu2026multimodal,
  title     = {Multimodal Attention Fusion for Audio-Visual Emotion Recognition},
  booktitle = {IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2026},
  month     = {December},
}
```

## Author

Zheyi Liu — National University of Singapore, Master of Artificial Intelligence Systems
