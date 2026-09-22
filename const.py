# Central configuration for the whole pipeline.
# Every script imports from here, so reported runs stay comparable.
#
# Methodology:
# - Validation split is by PATIENT, not random. Windows overlap by 83%, so a
#   random split would put near-copies of the same signal on both sides.
# - 3 s windows with 0.5 s hop trade independence for count: 366 windows come
#   from only 29 recordings of 2 patients.
from pathlib import Path

ROOT     = Path(__file__).parent
SRC_DIR  = ROOT / "brain_layer_annotations"
DATA_DIR = ROOT / "dataset"
WAV_DIR  = DATA_DIR / "wavs"
MANIFEST = DATA_DIR / "manifest.csv"

MODEL_DIR           = ROOT / "models"
BASELINE_PT         = MODEL_DIR / "baseline.pt"

BASELINE_METRICS    = ROOT / "baseline_metrics.json"
BASELINE_CURVES_PNG = ROOT / "baseline_curves.png"
PRUNED_METRICS      = ROOT / "pruned_metrics.json"
PRUNED_CURVES_PNG   = ROOT / "pruned_curves.png"
QUANT_METRICS       = ROOT / "quant_fp16_metrics.json"
ABLATION_METRICS    = ROOT / "ablation_metrics.json"
AUGMENT_METRICS     = ROOT / "augment_dropout_metrics.json"

MODEL_ID = "microsoft/wavlm-base-plus"

SR             = 16_000
WIN_SEC        = 3.0
WINDOW_SAMPLES = SR * WIN_SEC
HOP_SEC        = 0.5
SEED           = 21
VAL_PATIENTS   = {"P106"}
N_CLASSES      = 3
EMB_DIM        = 768

CLASS_MAP = {
    "External Globus Pallidus": "GPe",
    "Internal Globus Pallidus": "GPi",
    "Striatum or Putamen":      "STR",
}
LABEL_IDX = {name: i for i, name in enumerate(sorted(CLASS_MAP.values()))}

BATCH_SIZE        = 4
EPOCHS            = 20
LR_HEAD           = 1e-3
LR_BACKBONE       = 5e-5
WEIGHT_DECAY      = 1e-3
N_UNFREEZE_LAYERS = 2

PRUNE_LEVELS    = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
FINETUNE_EPOCHS = 7

# Regularisation settings shared by stages 4-6.
SPARSITY  = 0.30
DROPOUT   = 0.1
NOISE_STD = 0.05
MASK_FRAC = 0.10
AUG_PROB  = 0.5
