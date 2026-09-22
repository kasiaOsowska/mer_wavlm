# Shared library: dataset, model, train/eval loops, measurement helpers and
# the pruning routine. Defines no pipeline step of its own - the numbered
# scripts import from here and drive the work.
#
# Methodology:
# - Backbone is frozen except the last N_UNFREEZE_LAYERS transformer layers;
#   the rest of WavLM stays at its pretrained weights.
# - Loss is class-weighted (inverse frequency) because GPe/GPi/STR are skewed.
# - Pooling is a plain mean over time, head is a single Linear - see its comment.
# - prune_and_recover keeps pruned weights at zero during fine-tuning by masking
#   gradients, and restores the epoch with the best val F1. That selection makes
#   the reported val score optimistic - treat it as an upper bound.
from __future__ import annotations

import csv
import gzip
import io
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import WavLMModel

from const import (
    AUG_PROB,
    BASELINE_PT,
    FINETUNE_EPOCHS,
    LR_BACKBONE,
    LR_HEAD,
    MASK_FRAC,
    MODEL_ID,
    N_CLASSES,
    N_UNFREEZE_LAYERS,
    NOISE_STD,
    ROOT,
    WEIGHT_DECAY,
    WINDOW_SAMPLES,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

class MERWindows(Dataset):
    def __init__(self, manifest_path: Path, split: str) -> None:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        r = self.rows[idx]
        wav, _ = sf.read(str(ROOT / r["wav_path"]), dtype="float32")
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        return torch.from_numpy(wav.astype(np.float32)), int(r["label_idx"])

class AugmentedWindows(Dataset):
    """Waveform augmentation applied per access, so each epoch sees fresh noise."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.base[idx]
        if random.random() < AUG_PROB:
            x = x + NOISE_STD * torch.randn_like(x)
        if random.random() < AUG_PROB:
            mask_len = int(MASK_FRAC * x.shape[0])
            start = random.randint(0, x.shape[0] - mask_len)
            x = x.clone()
            x[start:start + mask_len] = 0.0
        return x, y

class BaselineModel(nn.Module):
    def __init__(self, backbone: WavLMModel, n_classes: int = N_CLASSES,
                 n_unfreeze_layers: int = N_UNFREEZE_LAYERS) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        if n_unfreeze_layers > 0:
            n_total = len(self.backbone.encoder.layers)
            k = min(n_unfreeze_layers, n_total)
            for layer in self.backbone.encoder.layers[n_total - k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        # linear probe - too little data (1 patient, ~19 recordings) for a deeper head
        self.head = nn.Linear(backbone.config.hidden_size, n_classes)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_values=input_values).last_hidden_state
        return self.head(hidden.mean(dim=1)) # mean-pool time

    def trainable_param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        backbone = [p for _, p in self.backbone.named_parameters() if p.requires_grad]
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone})
        groups.append({"params": list(self.head.parameters()), "lr": lr_head})
        return groups


def build_model() -> BaselineModel:
    return BaselineModel(WavLMModel.from_pretrained(MODEL_ID)).to(DEVICE)

def class_weighted_loss(train_rows: list[dict]) -> nn.CrossEntropyLoss:
    labels = np.array([int(r["label_idx"]) for r in train_rows])
    counts = np.bincount(labels, minlength=N_CLASSES)
    w = torch.tensor(len(labels) / (N_CLASSES * np.maximum(counts, 1)),
                     dtype=torch.float32, device=DEVICE)
    return nn.CrossEntropyLoss(weight=w)

def train_one_epoch(model: nn.Module, loader, opt: torch.optim.Optimizer,
                    loss_fn: nn.Module) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: nn.Module = None,
             half: bool = False) -> dict:
    model.eval()
    y_true, y_pred = [], []
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if half:
            x = x.half()
        logits = model(x)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(y.cpu().tolist())
        if loss_fn is not None:
            total_loss += loss_fn(logits, y).item() * x.size(0)
            total_n    += x.size(0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(N_CLASSES))).tolist(),
    }
    if loss_fn is not None:
        out["loss"] = total_loss / max(total_n, 1)
    return out


# Measurements shared by stages 4 and 5
def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def measure_inference_latency(model: nn.Module, n_iters: int = 30,
                              half: bool = False):
    model.eval()
    dummy = torch.randn(1, int(WINDOW_SAMPLES), device=DEVICE) * 0.1
    if half:
        dummy = dummy.half()
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(dummy)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return 1000.0 * dt / n_iters


def peak_memory_mb():
    out = {"ram_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)}
    if DEVICE.type == "cuda":
        out["vram_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return out

def count_nonzero_params(model: nn.Module):
    return int(sum((p != 0).sum().item() for p in model.parameters()))


def serialized_sizes_mb(model: nn.Module):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getbuffer().nbytes
    gz = len(gzip.compress(buf.getvalue(), compresslevel=6))
    return raw / (1024 * 1024), gz / (1024 * 1024)


def load_baseline_model() -> BaselineModel:
    model = build_model()
    ckpt = torch.load(BASELINE_PT, map_location=DEVICE)
    model.head.load_state_dict(ckpt["head_state"])
    return model


def measure_model(model: nn.Module, val_dl) -> dict:
    val = evaluate(model, val_dl)
    raw_mb, gz_mb = serialized_sizes_mb(model)
    return {
        "f1_macro": val["f1_macro"],
        "accuracy": val["accuracy"],
        "confusion": val["confusion"],
        "nonzero_params": count_nonzero_params(model),
        "size_raw_mb": raw_mb,
        "size_gzip_mb": gz_mb,
        "inference_ms": measure_inference_latency(model),
    }


# Unstructured pruning + fine-tuning that keeps zeros (stages 5-7)
def get_prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    targets = []
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            continue
        if parametrize.is_parametrized(module, "weight"):
            continue
        targets.append((module, "weight"))
    return targets


def real_sparsity(modules: list[tuple[nn.Module, str]]) -> float:
    zeros = 0
    total = 0
    for module, name in modules:
        weight = getattr(module, name)
        zeros += int((weight == 0).sum())
        total += weight.numel()
    return zeros / total


def set_finetune_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    layers = model.backbone.encoder.layers
    keep = min(N_UNFREEZE_LAYERS, len(layers))
    start = len(layers) - keep
    for layer in layers[start:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def make_mask_hook(mask: torch.Tensor):
    def hook(grad):
        return grad * mask
    return hook


def keep_zeros_during_finetune(targets: list[tuple[nn.Module, str]]) -> list:
    hooks = []
    for module, name in targets:
        weight = getattr(module, name)
        if not weight.requires_grad:
            continue
        mask = (weight != 0).to(weight.dtype)
        hook = weight.register_hook(make_mask_hook(mask))
        hooks.append(hook)
    return hooks


def clone_trainable(model: nn.Module) -> dict:
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            state[name] = p.detach().clone()
    return state


def restore_trainable(model: nn.Module, state: dict) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in state:
                p.copy_(state[name])


def prune_and_recover(model: nn.Module, sparsity_level: float, train_dl, val_dl,
                      train_rows: list[dict],
                      weight_decay: float = WEIGHT_DECAY) -> float:
    """Prune, then fine-tune. sparsity_level <= 0 fine-tunes without pruning."""
    targets = get_prunable_modules(model)
    hooks = []
    if sparsity_level > 0:
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured,
                                  amount=sparsity_level)
        for module, name in targets:
            prune.remove(module, name)
    sparsity = real_sparsity(targets)

    set_finetune_grads(model)
    if sparsity_level > 0:
        hooks = keep_zeros_during_finetune(targets)
    optimizer = torch.optim.AdamW(model.trainable_param_groups(LR_BACKBONE, LR_HEAD),
                                  weight_decay=weight_decay)
    loss_fn = class_weighted_loss(train_rows)

    best_f1 = -1.0
    best_state = None
    for _ in range(FINETUNE_EPOCHS):
        train_one_epoch(model, train_dl, optimizer, loss_fn)
        metrics = evaluate(model, val_dl)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = clone_trainable(model)

    if best_state is not None:
        restore_trainable(model, best_state)
    for hook in hooks:
        hook.remove()
    return sparsity
