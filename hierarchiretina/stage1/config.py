"""Stage I configuration presets.

``convnextv2_768`` is the deployed screening gate reported in the paper. The other presets are
the baselines of Table ``tab:stage1`` (SwinV2-L at 384 px, ConvNeXt V2-L at 512 px as a single
model and as five fold models). Every value below is copied from the notebook that produced the
corresponding checkpoint; where a notebook and the manuscript disagree, the notebook value is
kept (see the repository report for the list).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

#: Validation-derived operating threshold of the deployed gate (full float precision).
#: Rounding it to 0.2392 misroutes one Grade-0 test image (19,153 vs 19,154 routed).
GATE_THRESHOLD: float = 0.2391715943813324

#: Rounded threshold used by the original equal-load comparison (Cell 5.D). Thresholding the
#: 768-px predictions at this value emits 19,153 positives, the routing load of Table tab:stage1.
EQUAL_LOAD_REFERENCE_THRESHOLD: float = 0.2392

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Stage1Config:
    """All settings of one Stage I training run (one notebook ``CFG`` class)."""

    name: str
    # -- model ---------------------------------------------------------------------------
    model_name: str
    img_size: int
    head_hidden: int                      # width of head.2 (1536 ConvNeXt, 768 SwinV2)
    dropout: float
    pass_img_size_to_backbone: bool       # SwinV2 needs img_size at construction
    seq_gem_learnable: bool               # GeM with learnable p on [B, N, C] features (SwinV2)
    resize_input: bool                    # bicubic-resize inputs of the wrong size (SwinV2)
    # -- preprocessing / augmentation ------------------------------------------------------
    preprocess: Literal["square_crop", "letterbox"]
    train_aug: Literal["gate", "baseline"]
    tta_rotate_limit: int
    # -- data split ------------------------------------------------------------------------
    split: Literal["grade_stratified", "binary_stratified", "kfold", "three_way"]
    val_split: float
    test_split: float = 0.0               # only for the three-way SwinV2 split
    n_folds: int = 5
    seed: int = 42
    # -- optimisation ----------------------------------------------------------------------
    epochs: int = 45
    batch_size: int = 8
    grad_accum: int = 6
    val_batch_size: int = 4
    lr: float = 8e-5                      # head (GeM + attention pool + MLP)
    lr_backbone: float = 8e-6
    weight_decay: float = 1.5e-2
    lr_min: float = 1e-6                  # cosine floor, applied as the ratio lr_min / lr
    warmup_epochs: int = 4
    grad_clip: float = 1.0
    ema_decay: float = 0.9998
    patience: int = 8                     # early stopping on validation AUC
    amp: bool = True
    channels_last: bool = True
    deterministic: bool = False           # cudnn.deterministic flag set by seed_everything
    reseed_each_epoch: bool = False       # seed_everything(seed + epoch) before every epoch
    # -- loss / sampling -------------------------------------------------------------------
    bce_weight: float = 0.4
    focal_weight: float = 0.6
    focal_gamma: float = 2.0
    focal_alpha: float = 0.6
    label_smoothing: float = 0.03
    pos_weight_x: float | None = 1.6      # None -> pos_weight fixed at 1.0
    mixup_alpha: float = 0.0              # applied to half of the batches when > 0
    grade1_weight: float = 1.0            # extra sampler weight for Grade-1 images
    # -- operating threshold ---------------------------------------------------------------
    threshold_rule: Literal["2tpr-fpr", "youden"] = "2tpr-fpr"
    threshold_clip: tuple[float, float] | None = (0.20, 0.60)
    log_metrics_at_threshold: bool = True  # False: per-epoch sens/spec/F1 logged at 0.5

    def with_overrides(self, **kwargs) -> "Stage1Config":
        """Return a copy with some fields replaced."""
        return replace(self, **kwargs)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
SWINV2_NAME = "swinv2_large_window12to24_192to384"

PRESETS: dict[str, Stage1Config] = {
    # Deployed gate: dr_convnextv2_768_grade1.ipynb
    "convnextv2_768": Stage1Config(
        name="convnextv2_768",
        model_name=CONVNEXT_NAME, img_size=768, head_hidden=1536, dropout=0.35,
        pass_img_size_to_backbone=False, seq_gem_learnable=False, resize_input=False,
        preprocess="square_crop", train_aug="gate", tta_rotate_limit=8,
        split="grade_stratified", val_split=0.20,
        epochs=45, batch_size=8, grad_accum=6, val_batch_size=4,
        lr=8e-5, lr_backbone=8e-6, weight_decay=1.5e-2, lr_min=1e-6, warmup_epochs=4,
        ema_decay=0.9998, patience=8, deterministic=False, reseed_each_epoch=True,
        bce_weight=0.4, focal_weight=0.6, focal_gamma=2.0, focal_alpha=0.6,
        label_smoothing=0.03, pos_weight_x=1.6, mixup_alpha=0.0, grade1_weight=2.5,
        threshold_rule="2tpr-fpr", threshold_clip=(0.20, 0.60), log_metrics_at_threshold=True,
    ),
    # Baseline: dr_convnextv2_pipeline.ipynb, Cells 1-5 (single model, "single_cnx")
    "convnextv2_512": Stage1Config(
        name="convnextv2_512",
        model_name=CONVNEXT_NAME, img_size=512, head_hidden=1536, dropout=0.40,
        pass_img_size_to_backbone=False, seq_gem_learnable=False, resize_input=False,
        preprocess="letterbox", train_aug="baseline", tta_rotate_limit=10,
        split="binary_stratified", val_split=0.15,
        epochs=60, batch_size=16, grad_accum=4, val_batch_size=32,
        lr=1e-4, lr_backbone=1e-5, weight_decay=2e-2, lr_min=1e-6, warmup_epochs=3,
        ema_decay=0.9998, patience=6, deterministic=True, reseed_each_epoch=False,
        bce_weight=0.4, focal_weight=0.6, focal_gamma=2.0, focal_alpha=0.6,
        label_smoothing=0.05, pos_weight_x=1.5, mixup_alpha=0.4, grade1_weight=1.0,
        threshold_rule="2tpr-fpr", threshold_clip=(0.30, 0.60), log_metrics_at_threshold=False,
    ),
    # Baseline: dr_convnextv2_pipeline.ipynb, Cell 5.B (five fold models "fold_1".."fold_5")
    "convnextv2_512_fold": Stage1Config(
        name="convnextv2_512_fold",
        model_name=CONVNEXT_NAME, img_size=512, head_hidden=1536, dropout=0.40,
        pass_img_size_to_backbone=False, seq_gem_learnable=False, resize_input=False,
        preprocess="letterbox", train_aug="baseline", tta_rotate_limit=10,
        split="kfold", val_split=0.20, n_folds=5,
        epochs=40, batch_size=16, grad_accum=4, val_batch_size=16,
        lr=1e-4, lr_backbone=1e-5, weight_decay=2e-2, lr_min=1e-6, warmup_epochs=3,
        ema_decay=0.9998, patience=4, deterministic=True, reseed_each_epoch=False,
        bce_weight=0.4, focal_weight=0.6, focal_gamma=2.0, focal_alpha=0.6,
        label_smoothing=0.05, pos_weight_x=1.6, mixup_alpha=0.4, grade1_weight=1.0,
        threshold_rule="2tpr-fpr", threshold_clip=(0.30, 0.60), log_metrics_at_threshold=False,
    ),
    # Baseline: dr_classification_pipeline_CLEAN.ipynb ("swinv2")
    "swinv2_384": Stage1Config(
        name="swinv2_384",
        model_name=SWINV2_NAME, img_size=384, head_hidden=768, dropout=0.35,
        pass_img_size_to_backbone=True, seq_gem_learnable=True, resize_input=True,
        preprocess="letterbox", train_aug="baseline", tta_rotate_limit=10,
        split="three_way", val_split=0.10, test_split=0.10,
        epochs=80, batch_size=8, grad_accum=8, val_batch_size=8,
        lr=8e-5, lr_backbone=8e-6, weight_decay=1.5e-2, lr_min=1e-6, warmup_epochs=4,
        ema_decay=0.9995, patience=5, deterministic=False, reseed_each_epoch=False,
        bce_weight=0.5, focal_weight=0.5, focal_gamma=2.0, focal_alpha=0.5,
        label_smoothing=0.05, pos_weight_x=None, mixup_alpha=0.2, grade1_weight=1.0,
        threshold_rule="youden", threshold_clip=None, log_metrics_at_threshold=False,
    ),
}

DEFAULT_PRESET = "convnextv2_768"


def get_preset(name: str = DEFAULT_PRESET, **overrides) -> Stage1Config:
    """Return a preset by name, optionally with fields overridden."""
    if name not in PRESETS:
        raise KeyError(f"Unknown Stage I preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[name]
    return cfg.with_overrides(**overrides) if overrides else cfg
