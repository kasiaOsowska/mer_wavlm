# Stage 1: turn raw CSV recordings into 16 kHz wav windows + dataset/manifest.csv.
# Run this first - every other script reads the manifest.
#
# Methodology:
# - Sample rate is inferred per file from the timestamp column, not assumed.
# - RMS normalisation to -23 dBFS happens BEFORE windowing, so loudness
#   differences between recordings cannot become a class cue.
# - Windows are cut with 83% overlap to raise the sample count; they are
#   therefore not independent observations.
# - Split is assigned by patient (VAL_PATIENTS), never by shuffling windows.
from __future__ import annotations

import csv
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from const import *

np.random.seed(SEED)
torch.manual_seed(SEED)

def build_label_map() -> dict[tuple[str, str], str]:
    label_map: dict[tuple[str, str], str] = {}
    duplicates: list[tuple[str, str]] = []

    for class_folder, label in CLASS_MAP.items():
        class_dir = SRC_DIR / class_folder
        for patient_dir in class_dir.iterdir():
            patient = patient_dir.name
            for csv_path in patient_dir.glob("*.csv"):
                key = (patient, csv_path.name)
                if key in label_map and label_map[key] != label:
                    duplicates.append(key)
                label_map[key] = label
    return label_map


def load_full_signal(csv_path: Path) -> tuple[np.ndarray, int]:
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1), dtype=np.float64)
    t   = arr[:, 0]
    sig = arr[:, 1].astype(np.float32)
    dt  = float(np.median(np.diff(t)))
    sr  = int(round(1.0 / dt))
    return sig, sr


def resample(sig: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return sig.astype(np.float32)
    t = torch.from_numpy(sig).unsqueeze(0)
    out = AF.resample(t, sr_in, sr_out).squeeze(0).numpy()
    return out.astype(np.float32)


def rms_normalize(sig: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(sig**2) + 1e-12))
    sig = sig * (10 ** (target_dbfs / 20) / rms)
    peak = float(np.max(np.abs(sig)) + 1e-12)
    if peak > 0.99:
        sig = sig * (0.99 / peak)
    return sig.astype(np.float32)


def make_chunks(sig: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> list[np.ndarray]:
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if len(sig) < win:
        out = np.zeros(win, dtype=np.float32)
        out[: len(sig)] = sig
        return [out]
    return [sig[s : s + win] for s in range(0, len(sig) - win + 1, hop)]


def parse_depth(stem: str) -> str:
    m = re.search(r"depth(-?\d+(?:,\d+)?)", stem)
    return m.group(1).replace(",", ".") if m else "NA"


WAV_DIR.mkdir(parents=True, exist_ok=True)

label_map = build_label_map()
print(f"Labels from class folders: {len(label_map)}")

rows: list[dict] = []
per_class_split: dict[tuple[str, str], int] = {}
detected_srs: list[int] = []
missing_full: list[str] = []

for (patient, fname), label in sorted(label_map.items()):
    full_path = SRC_DIR / patient / fname
    if not full_path.is_file():
        missing_full.append(f"{patient}/{fname}")
        continue

    depth = parse_depth(Path(fname).stem)

    sig, sr_in = load_full_signal(full_path)
    detected_srs.append(sr_in)
    sig = resample(sig, sr_in, SR)
    sig = rms_normalize(sig)
    chunks = make_chunks(sig, SR, WIN_SEC, HOP_SEC)

    out_dir = WAV_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(chunks):
        out_fname = f"{patient}_d{depth}_c{i:03d}.wav"
        out_path = out_dir / out_fname
        sf.write(str(out_path), c, SR, subtype="PCM_16")
        rows.append({
            "wav_path":  str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "label":     label,
            "label_idx": LABEL_IDX[label],
            "patient":   patient,
            "depth":     depth,
            "chunk":     i,
            "split":     "TBD",
        })

    dur = len(sig) / SR
    print(f"  {patient}/{fname:35s}  sr_in={sr_in:>6}Hz  {dur:6.2f}s  ->  {len(chunks):3d} windows   [{label}]")

for r in rows:
    r["split"] = "val" if r["patient"] in VAL_PATIENTS else "train"

for r in rows:
    per_class_split[(r["label"], r["split"])] = per_class_split.get((r["label"], r["split"]), 0) + 1

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)