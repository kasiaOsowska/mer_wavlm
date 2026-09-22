# Stage 5: the full regularisation pipeline - waveform augmentation + dropout
# before the head + pruning + fp16. Produces augment_dropout_metrics.json.
#
# Methodology:
# - Augmentation is applied in __getitem__, so every epoch sees different noise
#   and masks rather than a fixed enlarged dataset.
# - Noise is scaled relative to the signal: MERWindows z-score normalises, so
#   NOISE_STD is a fraction of the signal's own standard deviation.
# - Training accuracy is measured on train_clean_dl (un-augmented) so the gap to
#   val accuracy is a readable overfitting signal.
# - There is no matching non-augmented control run in the project, so this
#   script alone cannot prove augmentation helped.
from __future__ import annotations

import json
import random

import torch.nn as nn
from torch.utils.data import DataLoader

from const import (
    AUGMENT_METRICS,
    AUG_PROB,
    BATCH_SIZE,
    DROPOUT,
    FINETUNE_EPOCHS,
    MANIFEST,
    MASK_FRAC,
    NOISE_STD,
    SEED,
    SPARSITY,
)
from core import (
    DEVICE,
    AugmentedWindows,
    MERWindows,
    evaluate,
    load_baseline_model,
    prune_and_recover,
    serialized_sizes_mb,
    set_seed,
)

set_seed(SEED)
random.seed(SEED)

train_ds = MERWindows(MANIFEST, split="train")
val_ds = MERWindows(MANIFEST, split="val")
train_aug = AugmentedWindows(train_ds)
train_dl = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_clean_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = load_baseline_model()
model.head = nn.Sequential(nn.Dropout(DROPOUT), model.head).to(DEVICE)
prune_and_recover(model, SPARSITY, train_dl, val_dl, train_ds.rows)
model.eval()

train_metrics = evaluate(model, train_clean_dl)
val_metrics = evaluate(model, val_dl)
raw_fp32, gz_fp32 = serialized_sizes_mb(model)

model = model.half()
val_metrics_h = evaluate(model, val_dl, half=True)
raw_fp16, gz_fp16 = serialized_sizes_mb(model)

print(f"Pipeline: augmentation (noise {NOISE_STD}, masking {MASK_FRAC:.0%}) + "
      f"dropout {DROPOUT} + pruning {SPARSITY:.0%} + fp16")
print(f"fp32  train_acc={train_metrics['accuracy']:.3f}  val_F1={val_metrics['f1_macro']:.3f}  "
      f"val_acc={val_metrics['accuracy']:.3f}  raw={raw_fp32:.0f}MB gzip={gz_fp32:.0f}MB")
print(f"fp16  val_F1={val_metrics_h['f1_macro']:.3f}  val_acc={val_metrics_h['accuracy']:.3f}  "
      f"raw={raw_fp16:.0f}MB gzip={gz_fp16:.0f}MB")

AUGMENT_METRICS.write_text(json.dumps({
    "config": {
        "seed": SEED, "batch_size": BATCH_SIZE, "sparsity": SPARSITY,
        "finetune_epochs": FINETUNE_EPOCHS, "dropout": DROPOUT,
        "noise_std": NOISE_STD, "mask_frac": MASK_FRAC, "aug_prob": AUG_PROB,
        "device": str(DEVICE),
    },
    "data": {"train_windows": len(train_ds), "val_windows": len(val_ds)},
    "train_clean": train_metrics,
    "fp32": {**val_metrics, "size_raw_mb": raw_fp32, "size_gzip_mb": gz_fp32},
    "fp16": {**val_metrics_h, "size_raw_mb": raw_fp16, "size_gzip_mb": gz_fp16},
}, indent=2), encoding="utf-8")
print(f"Metrics saved to {AUGMENT_METRICS.name}")
