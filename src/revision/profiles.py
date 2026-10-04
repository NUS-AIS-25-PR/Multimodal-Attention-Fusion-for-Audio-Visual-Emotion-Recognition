"""Canonical production profiles; smoke overrides are explicit and separate."""
from __future__ import annotations

from train import build_arg_parser

GATED = {
    "epochs": 30, "batch_size": 8, "lr": 3e-4, "weight_decay": 1e-4,
    "two_stage_training": True, "stage1_epochs": 5,
    "audio_backbone_lr": 1e-5, "video_backbone_lr": 1e-5,
    "fusion_unfreeze_wavlm_layers": 2, "fusion_unfreeze_video_blocks": 1,
    "use_cosine_annealing": True, "early_stopping_patience": 8,
}
OVERRIDES = {
    "audio": {"epochs": 20, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
              "wavlm_stage": 2, "backbone_lr": 3e-5,
              "use_cosine_annealing": True, "early_stopping_patience": 10},
    "video": {"epochs": 20, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
              "use_cosine_annealing": True, "early_stopping_patience": 10},
    "gated": GATED,
    "concat": GATED,
    "chumachenko_ia": GATED,
    "xattn": {
        "epochs": 35, "batch_size": 8, "lr": 2e-4, "weight_decay": 2e-4,
        "xattn_head": "gated", "xattn_d_model": 96, "xattn_heads": 4,
        "xattn_attn_dropout": 0.1, "xattn_stochastic_depth": 0.1,
        "label_smoothing": 0.05, "two_stage_training": True, "stage1_epochs": 6,
        "audio_backbone_lr": 8e-6, "video_backbone_lr": 8e-6,
        "fusion_unfreeze_wavlm_layers": 2, "fusion_unfreeze_video_blocks": 1,
        "use_cosine_annealing": True, "early_stopping_patience": 10,
    },
}
# These identify fold-specific files/actors, rather than the model profile.
RUN_FIELDS = {"data_root", "train_actors", "val_actors", "test_actors", "audio_ckpt", "video_ckpt"}


def production_profile(model: str) -> dict:
    if model not in OVERRIDES:
        raise ValueError(f"Unknown production model: {model}")
    defaults = vars(build_arg_parser().parse_args(["--data_root", "unused"]))
    config = {k: v for k, v in defaults.items() if k not in RUN_FIELDS}
    config.update(fusion=model, split_mode="actor", use_wavlm=True, num_classes=8,
                  num_workers=-1, smoke=False)
    config.update(OVERRIDES[model])
    return {"id": f"spmb2026-{model}-v1", "config": config}


def resolve_profiles(smoke: bool, settings: dict) -> dict:
    profiles = {}
    for model in OVERRIDES:
        profile = production_profile(model)
        config = profile["config"]
        config.update(seed=settings["seed"], frames=settings["frames"],
                      use_face_crop=settings["use_face_crop"], num_workers=settings["num_workers"])
        if smoke:
            profile["id"] = f"spmb2026-{model}-smoke-v1"
            config.update(smoke=True, epochs=2, batch_size=4, frames=2,
                          num_workers=0, use_face_crop=False, no_pretrained_video=True)
            if config["two_stage_training"]:
                config["stage1_epochs"] = 1
        profiles[model] = profile
    return profiles
