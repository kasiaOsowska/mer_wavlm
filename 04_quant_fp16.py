# Stage 4: one operating point (SPARSITY) converted to float16, comparing
# quality, size and latency against fp32. Produces quant_fp16_metrics.json.
#
# Methodology:
# - fp32 metrics are taken BEFORE .half(), which converts the model in place.
# - Latency uses warm-up iterations and cuda.synchronize, otherwise the timing
#   would capture kernel launch and compilation rather than inference.
# - Same 30% sparsity as stage 5, so the two runs differ only by regularisation.
from __future__ import annotations

import json

from torch.utils.data import DataLoader

from const import (
    BATCH_SIZE,
    FINETUNE_EPOCHS,
    MANIFEST,
    QUANT_METRICS,
    SEED,
    SPARSITY,
)
from core import (
    DEVICE,
    MERWindows,
    evaluate,
    load_baseline_model,
    measure_inference_latency,
    prune_and_recover,
    serialized_sizes_mb,
    set_seed,
)


set_seed(SEED)

train_ds = MERWindows(MANIFEST, split="train")
val_ds = MERWindows(MANIFEST, split="val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = load_baseline_model()
prune_and_recover(model, SPARSITY, train_dl, val_dl, train_ds.rows)
model.eval()

metrics_fp32 = evaluate(model, val_dl)
raw_fp32, gz_fp32 = serialized_sizes_mb(model)
lat_fp32 = measure_inference_latency(model)

model = model.half()

metrics_fp16 = evaluate(model, val_dl, half=True)
raw_fp16, gz_fp16 = serialized_sizes_mb(model)
lat_fp16 = measure_inference_latency(model, half=True)

print(f"Model at {SPARSITY:.0%} sparsity, float16 conversion, measured on {DEVICE}:")
print(f"{'':5} | {'F1':>6} | {'acc':>6} | {'raw MB':>7} | {'gzip MB':>7} | {'lat ms':>7}")
print(f"{'fp32':5} | {metrics_fp32['f1_macro']:6.3f} | {metrics_fp32['accuracy']:6.3f} | "
      f"{raw_fp32:7.1f} | {gz_fp32:7.1f} | {lat_fp32:7.1f}")
print(f"{'fp16':5} | {metrics_fp16['f1_macro']:6.3f} | {metrics_fp16['accuracy']:6.3f} | "
      f"{raw_fp16:7.1f} | {gz_fp16:7.1f} | {lat_fp16:7.1f}")

QUANT_METRICS.write_text(json.dumps({
    "config": {
        "seed": SEED, "batch_size": BATCH_SIZE, "sparsity": SPARSITY,
        "finetune_epochs": FINETUNE_EPOCHS, "device": str(DEVICE),
    },
    "data": {"train_windows": len(train_ds), "val_windows": len(val_ds)},
    "fp32": {**metrics_fp32, "size_raw_mb": raw_fp32,
             "size_gzip_mb": gz_fp32, "inference_ms": lat_fp32},
    "fp16": {**metrics_fp16, "size_raw_mb": raw_fp16,
             "size_gzip_mb": gz_fp16, "inference_ms": lat_fp16},
}, indent=2), encoding="utf-8")
print(f"Metrics saved to {QUANT_METRICS.name}")
