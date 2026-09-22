# Stage 2: train the reference model. Produces models/baseline.pt,
# baseline_metrics.json and baseline_curves.png. Stages 3-6 all start from it.
#
# Methodology:
# - Checkpoint is chosen by best val macro-F1, not accuracy, because the classes
#   are imbalanced and accuracy would favour the majority class.
# - Two learning rates: a small one for the unfrozen backbone layers, a larger
#   one for the freshly initialised head.
# - Only head_state is saved. The fine-tuned transformer layers are NOT
#   persisted, so later stages reload a pretrained backbone plus this head.
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import torchinfo

from const import (
    BASELINE_CURVES_PNG,
    BASELINE_METRICS,
    BASELINE_PT,
    BATCH_SIZE,
    EMB_DIM,
    EPOCHS,
    LR_BACKBONE,
    LR_HEAD,
    MANIFEST,
    MODEL_DIR,
    MODEL_ID,
    N_CLASSES,
    N_UNFREEZE_LAYERS,
    SEED,
    WEIGHT_DECAY,
)
from core import (
    DEVICE,
    MERWindows,
    build_model,
    class_weighted_loss,
    count_params,
    evaluate,
    measure_inference_latency,
    model_size_mb,
    peak_memory_mb,
    set_seed,
    train_one_epoch,
)


set_seed(SEED)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")

train_ds = MERWindows(MANIFEST, split="train")
val_ds   = MERWindows(MANIFEST, split="val")
print(f"Train: {len(train_ds)} windows  Val: {len(val_ds)} windows")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = build_model()
total, trainable = count_params(model)
print(f"Parameters  total: {total:,}   trainable: {trainable:,}")

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

opt = torch.optim.AdamW(
    model.trainable_param_groups(lr_backbone=LR_BACKBONE, lr_head=LR_HEAD),
    weight_decay=WEIGHT_DECAY,
)
loss_fn = class_weighted_loss(train_ds.rows)
print(f"lr_backbone={LR_BACKBONE}  lr_head={LR_HEAD}  "
      f"unfrozen transformer layers: {N_UNFREEZE_LAYERS}")

print("Training start")
t_start = time.perf_counter()
history, best_f1 = [], -1.0

torchinfo.summary(model, input_size=(1, 48000))

for epoch in range(1, EPOCHS + 1):
    t_epoch = time.perf_counter()
    train_loss = train_one_epoch(model, train_dl, opt, loss_fn)
    epoch_sec = time.perf_counter() - t_epoch

    train_metrics = evaluate(model, train_dl, loss_fn=loss_fn)
    val_metrics   = evaluate(model, val_dl,   loss_fn=loss_fn)
    history.append({
        "epoch": epoch, "train_loss": train_loss, "epoch_sec": epoch_sec,
        "train_acc": train_metrics["accuracy"],
        "val_loss":  val_metrics["loss"],
        "accuracy":  val_metrics["accuracy"],
        "f1_macro":  val_metrics["f1_macro"],
    })
    print(f"Ep {epoch:02d}  trL={train_loss:.4f}  trA={train_metrics['accuracy']:.3f}  "
          f"vL={val_metrics['loss']:.4f}  vA={val_metrics['accuracy']:.3f}  "
          f"vF1={val_metrics['f1_macro']:.3f}  ({epoch_sec:.1f}s)")

    if val_metrics["f1_macro"] > best_f1:
        best_f1 = val_metrics["f1_macro"]
        torch.save({"head_state": model.head.state_dict(),
                    "n_classes": N_CLASSES, "emb_dim": EMB_DIM,
                    "model_id": MODEL_ID}, BASELINE_PT)

total_train_sec = time.perf_counter() - t_start
mean_epoch_sec  = float(np.mean([h["epoch_sec"] for h in history]))

final_val  = evaluate(model, val_dl)
latency_ms = measure_inference_latency(model)
mem        = peak_memory_mb()
size_mb    = model_size_mb(BASELINE_PT)

print("=========== BASELINE SUMMARY ===========")
print(f"Training time / epoch : {mean_epoch_sec:.2f} s")
print(f"Inference time / sample : {latency_ms:.2f} ms")
print(f"Best val F1 (macro) : {best_f1:.3f}")
print(f"Final val accuracy : {final_val['accuracy']:.3f}")
print("Confusion matrix (val) :")
for row in final_val["confusion"]:
    print(f"  {row}")
print(f"Saved head size : {size_mb:.3f} MB")
print(f"Peak RAM : {mem['ram_mb']:.0f} MB")
if "vram_mb" in mem:
    print(f"Peak VRAM  : {mem['vram_mb']:.0f} MB")

BASELINE_METRICS.write_text(json.dumps({
    "config": {
        "model_id": MODEL_ID, "seed": SEED, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr_head": LR_HEAD, "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY, "n_unfreeze_layers": N_UNFREEZE_LAYERS,
        "n_classes": N_CLASSES, "device": str(DEVICE),
    },
    "data": {"train_windows": len(train_ds), "val_windows": len(val_ds)},
    "params": {"total": total, "trainable": trainable},
    "history": history,
    "best_f1_macro": best_f1,
    "final_val": final_val,
    "mean_epoch_sec": mean_epoch_sec,
    "total_train_sec": total_train_sec,
    "inference_ms": latency_ms,
    "head_size_mb": size_mb,
    "peak_memory_mb": mem,
}, indent=2), encoding="utf-8")
print(f"Metrics saved to {BASELINE_METRICS.name}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs = [h["epoch"] for h in history]
fig, (axL, axA) = plt.subplots(1, 2, figsize=(11, 4))
axL.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
axL.plot(epochs, [h["val_loss"]   for h in history], "s-", label="val")
axL.set_xlabel("epoch"); axL.set_ylabel("Loss"); axL.set_title("Loss")
axL.grid(alpha=0.3); axL.legend()
axA.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
axA.plot(epochs, [h["accuracy"]  for h in history], "s-", label="val")
axA.set_xlabel("Epochs"); axA.set_ylabel("Accuracy"); axA.set_title("Accuracy")
axA.set_ylim(0, 1); axA.grid(alpha=0.3); axA.legend()
fig.suptitle(f"Baseline  N_UNFREEZE_LAYERS={N_UNFREEZE_LAYERS}  EPOCHS={EPOCHS}"
             f"  lr_head={LR_HEAD}  lr_bb={LR_BACKBONE}")
fig.tight_layout()
fig.savefig(BASELINE_CURVES_PNG, dpi=120)