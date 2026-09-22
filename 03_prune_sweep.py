# Stage 3: sweep global unstructured L1 pruning over PRUNE_LEVELS and record how
# F1 and model size trade off. Produces pruned_metrics.json and pruned_curves.png.
#
# Methodology:
# - Every level restarts from a fresh baseline rather than pruning the previous
#   model further, so the levels are directly comparable.
# - Sparsity is re-measured after prune.remove instead of trusting the requested
#   amount; both values are stored.
# - Size is reported after gzip, because zeroed weights still occupy full floats
#   on disk and only pay off once compressed.
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from const import (
    BATCH_SIZE,
    FINETUNE_EPOCHS,
    MANIFEST,
    PRUNE_LEVELS,
    PRUNED_CURVES_PNG,
    PRUNED_METRICS,
    SEED,
)
from core import (
    DEVICE,
    MERWindows,
    load_baseline_model,
    measure_model,
    prune_and_recover,
    set_seed,
)


set_seed(SEED)
print(f"Device: {DEVICE}")

train_ds = MERWindows(MANIFEST, split="train")
val_ds = MERWindows(MANIFEST, split="val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} windows   Val: {len(val_ds)} windows")

records = []

model = load_baseline_model()
measured = measure_model(model, val_dl)
records.append({"requested_sparsity": 0.0, "real_sparsity": 0.0, **measured})
print(f"reference 0%: F1={measured['f1_macro']:.3f} acc={measured['accuracy']:.3f} "
      f"gzip={measured['size_gzip_mb']:.0f}MB")
del model

for sparsity_level in PRUNE_LEVELS:
    model = load_baseline_model()
    sparsity = prune_and_recover(model, sparsity_level, train_dl, val_dl, train_ds.rows)
    measured = measure_model(model, val_dl)
    records.append({"requested_sparsity": sparsity_level,
                    "real_sparsity": sparsity, **measured})
    print(f"sparsity {sparsity:.0%}: F1={measured['f1_macro']:.3f} acc={measured['accuracy']:.3f} "
          f"gzip={measured['size_gzip_mb']:.0f}MB nonzero={measured['nonzero_params']:,}")
    del model

PRUNED_METRICS.write_text(json.dumps({
    "config": {
        "seed": SEED, "batch_size": BATCH_SIZE, "prune_levels": PRUNE_LEVELS,
        "finetune_epochs": FINETUNE_EPOCHS, "device": str(DEVICE),
    },
    "data": {"train_windows": len(train_ds), "val_windows": len(val_ds)},
    "levels": records,
}, indent=2), encoding="utf-8")
print(f"Metrics saved to {PRUNED_METRICS.name}")

sparsities = [r["real_sparsity"] for r in records]
f1_scores = [r["f1_macro"] for r in records]
sizes = [r["size_gzip_mb"] for r in records]

figure, axis_f1 = plt.subplots(figsize=(7, 4))
axis_f1.plot(sparsities, f1_scores, "o-", color="tab:blue", label="F1 macro")
axis_f1.set_xlabel("sparsity")
axis_f1.set_ylabel("F1 macro")
axis_f1.set_ylim(0, 1)
axis_size = axis_f1.twinx()
axis_size.plot(sparsities, sizes, "s--", color="tab:red", label="gzip size [MB]")
axis_size.set_ylabel("gzip size [MB]")
figure.suptitle("F1 and size vs sparsity")
figure.tight_layout()
figure.savefig(PRUNED_CURVES_PNG, dpi=120)
