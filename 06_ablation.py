# Stage 6: ablation. Runs one fine-tuning per regularisation technique, in isolation
# and combined, so each one's contribution can be read separately. Produces
# ablation_metrics.json. Stages 4-5 mix the techniques, this one does not.
#
# Methodology:
# - Every variant restarts from the same models/baseline.pt and reseeds before the
#   run, so variants differ only by the technique under test.
# - "none" is the control: no dropout, no augmentation, weight_decay=0, no pruning.
#   It still fine-tunes for FINETUNE_EPOCHS, so the comparison isolates regularisation
#   rather than the extra training.
# - Train accuracy is measured on un-augmented data, so train-val gap is comparable
#   across variants; that gap, not peak accuracy, is what regularisation should shrink.
# - prune_and_recover keeps the best-val-F1 epoch, so every reported val score is
#   optimistic in the same way. Compare variants to each other, not to absolute truth.

from __future__ import annotations

import json
import random
import time

import torch.nn as nn
from torch.utils.data import DataLoader

from const import (
    ABLATION_METRICS,
    AUG_PROB,
    BATCH_SIZE,
    DROPOUT,
    FINETUNE_EPOCHS,
    MANIFEST,
    MASK_FRAC,
    NOISE_STD,
    SEED,
    SPARSITY,
    WEIGHT_DECAY,
)
from core import (
    DEVICE,
    AugmentedWindows,
    MERWindows,
    count_nonzero_params,
    evaluate,
    load_baseline_model,
    prune_and_recover,
    serialized_sizes_mb,
    set_seed,
)

# name -> (dropout, augment, weight_decay, sparsity)
VARIANTS = [
    ("none",         0.0,     False, 0.0,          0.0),
    ("dropout",      DROPOUT, False, 0.0,          0.0),
    ("augment",      0.0,     True,  0.0,          0.0),
    ("weight_decay", 0.0,     False, WEIGHT_DECAY, 0.0),
    ("prune",        0.0,     False, 0.0,          SPARSITY),
    ("all",          DROPOUT, True,  WEIGHT_DECAY, SPARSITY),
]


def run_variant(name, dropout, augment, weight_decay, sparsity,
                train_ds, val_ds, val_dl, train_clean_dl):
    set_seed(SEED)
    random.seed(SEED)

    source = AugmentedWindows(train_ds) if augment else train_ds
    train_dl = DataLoader(source, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = load_baseline_model()
    if dropout > 0:
        model.head = nn.Sequential(nn.Dropout(dropout), model.head).to(DEVICE)

    t0 = time.perf_counter()
    real_sparsity = prune_and_recover(model, sparsity, train_dl, val_dl,
                                      train_ds.rows, weight_decay=weight_decay)
    elapsed = time.perf_counter() - t0

    model.eval()
    train_metrics = evaluate(model, train_clean_dl)
    val_metrics = evaluate(model, val_dl)
    raw_mb, gz_mb = serialized_sizes_mb(model)

    return {
        "variant": name,
        "settings": {"dropout": dropout, "augment": augment,
                     "weight_decay": weight_decay, "sparsity": sparsity},
        "real_sparsity": real_sparsity,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "gap": train_metrics["accuracy"] - val_metrics["accuracy"],
        "confusion": val_metrics["confusion"],
        "nonzero_params": count_nonzero_params(model),
        "size_raw_mb": raw_mb,
        "size_gzip_mb": gz_mb,
        "finetune_sec": elapsed,
    }


print(f"Device: {DEVICE}")
train_ds = MERWindows(MANIFEST, split="train")
val_ds = MERWindows(MANIFEST, split="val")
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_clean_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} windows   Val: {len(val_ds)} windows")
print(f"{len(VARIANTS)} variants x {FINETUNE_EPOCHS} epochs, all from models/baseline.pt\n")

records = []
for name, dropout, augment, weight_decay, sparsity in VARIANTS:
    print(f"-> {name}  (dropout={dropout} augment={augment} "
          f"wd={weight_decay} sparsity={sparsity:.0%})")
    record = run_variant(name, dropout, augment, weight_decay, sparsity,
                         train_ds, val_ds, val_dl, train_clean_dl)
    records.append(record)
    print(f"   train_acc={record['train_accuracy']:.3f}  val_acc={record['val_accuracy']:.3f}  "
          f"val_F1={record['val_f1_macro']:.3f}  gap={record['gap']:+.3f}  "
          f"({record['finetune_sec']:.0f}s)\n")

print("=========== ABLATION SUMMARY ===========")
header = f"{'variant':<13} | {'train acc':>9} | {'val acc':>7} | {'val F1':>6} | {'gap':>6} | {'gzip MB':>7}"
print(header)
print("-" * len(header))
for r in records:
    print(f"{r['variant']:<13} | {r['train_accuracy']:9.3f} | {r['val_accuracy']:7.3f} | "
          f"{r['val_f1_macro']:6.3f} | {r['gap']:+6.3f} | {r['size_gzip_mb']:7.1f}")

baseline = records[0]
print("\nChange vs the unregularised control:")
for r in records[1:]:
    print(f"  {r['variant']:<13} val F1 {r['val_f1_macro'] - baseline['val_f1_macro']:+.3f}   "
          f"gap {r['gap'] - baseline['gap']:+.3f}")

ABLATION_METRICS.write_text(json.dumps({
    "config": {
        "seed": SEED, "batch_size": BATCH_SIZE, "finetune_epochs": FINETUNE_EPOCHS,
        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY, "sparsity": SPARSITY,
        "noise_std": NOISE_STD, "mask_frac": MASK_FRAC, "aug_prob": AUG_PROB,
        "device": str(DEVICE),
    },
    "data": {"train_windows": len(train_ds), "val_windows": len(val_ds)},
    "variants": records,
}, indent=2), encoding="utf-8")
print(f"\nMetrics saved to {ABLATION_METRICS.name}")
