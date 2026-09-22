# WavLM on MER signals

Adapting a self-supervised **speech** model to classify basal ganglia structures from
intraoperative microelectrode recordings, then compressing it to fit an embedded target.

- Base model: [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
- MER data: [doi.org/10.34808/nq8v-t162](https://doi.org/10.34808/nq8v-t162)
- Data descriptor: Osowska, Szymański & Libionka (2026), *A Dataset of Microelectrode
  Recordings from Deep Brain Stimulation Procedures*, Scientific Data 13:870,
  [doi.org/10.1038/s41597-026-07492-w](https://doi.org/10.1038/s41597-026-07492-w)

## The question

During deep brain stimulation (DBS) surgery the microelectrode is advanced millimetre by
millimetre and the neurosurgeon localises subcortical structures partly by listening to the
signal. Automating that leads to two questions:

**1. Does a model pretrained on speech transfer to MER?**
The argument is spectral: most speech energy sits between ~100 Hz and a few kHz, MER sits at
roughly 500 to 5000 Hz, so both are fully represented at a 16 kHz sampling rate. There is no
public model pretrained directly on MER, so a speech encoder is the available starting point.
The analogy is deliberately shallow: speech is acoustic, MER is electrophysiological.

**2. Can the result run on the target hardware?**
The deployment target is the inomed ISIS MER system: an embedded computer, no server GPU,
limited memory. WavLM-base-plus is 94.4 M parameters, ~360 MB in fp32.

Concrete task: label a 3-second window as **GPe**, **GPi** or **STR** (striatum/putamen).

## Approach

**Data.** MER recordings (20 kHz) are resampled to 16 kHz, RMS-normalised to -23 dBFS, and cut
into 3 s windows with 0.5 s hop. That yields 366 windows across 3 classes. The split is by
patient (train on P121 with 280 windows, validate on P106 with 86) because the 83 % overlap
would make a random split leak near-identical signal into validation.

**Model.** WavLM-base-plus as a frozen backbone except the last 2 encoder layers; the X-vector
head of the `-sv` variant was dropped because it encoded patient identity rather than anatomy.
Instead: mean-pool the transformer output over time to a 768-dim vector, then a single `Linear(768, 3)`.
14.18 M of 94.4 M parameters are trained. Loss is class-weighted cross-entropy (the classes are
skewed), optimiser AdamW.

**Optimisation, three axes.** Global unstructured L1 pruning with 7-epoch recovery, swept from
30 % to 99 % sparsity; float16 conversion; and regularisation (waveform augmentation + dropout).
Each is measured against the unoptimised reference.

## Results

Validation patient P106, RTX 5070 Ti 16 GB.

**Reference model.** Strong overfitting, as expected from a single-patient training set:

| train acc | val acc | val F1 macro | size | latency |
|---|---|---|---|---|
| 0.911 | 0.510 | 0.510 | 360 MB | 7.8 to 9.1 ms |

**Pruning sweep.** Moderate pruning acts as regularisation and is the single best change:

| sparsity | F1 macro | accuracy | non-zero params | gzip |
|---|---|---|---|---|
| 0 % | 0.436 | 0.419 | 94 384 195 | 213.4 MB |
| **30 %** | **0.613** | **0.663** | 67 523 518 | 184.7 MB |
| 50 % | 0.395 | 0.477 | 49 616 369 | 148.6 MB |
| 70 % | 0.193 | 0.407 | 31 709 220 | 109.3 MB |
| ≥ 90 % | 0.193 | 0.407 | ≤ 13 802 071 | ≤ 63.1 MB |

Above 70 % the model collapses to predicting one class (F1 0.193).

**float16** on the 30 % model. Half the raw size, identical quality, no speedup:

| precision | F1 | accuracy | raw | gzip | latency |
|---|---|---|---|---|---|
| fp32 | 0.567 | 0.651 | 360.1 MB | 184.7 MB | 8.1 ms |
| fp16 | 0.567 | 0.651 | 180.1 MB | 139.2 MB | 8.4 ms |

**Augmentation + dropout** on the 30 % model. Removes the overfitting gap without raising peak
accuracy (validation now exceeds training, because training is augmented and validation is clean):

| model | train acc | val acc | F1 macro |
|---|---|---|---|
| reference | 0.911 | 0.510 | 0.510 |
| 30 % + augmentation + dropout | 0.582 | 0.628 | 0.545 |

**Ablation**. Each technique in isolation,
all six variants starting from the same checkpoint and fine-tuned for the same 7 epochs:

| variant | train acc | val acc | val F1 | gap | gzip |
|---|---|---|---|---|---|
| none (control) | 0.575 | 0.640 | **0.609** | -0.065 | 231.7 MB |
| dropout | 0.586 | 0.477 | 0.531 | +0.109 | 231.7 MB |
| augmentation | 0.714 | 0.535 | 0.508 | +0.179 | 231.7 MB |
| weight decay | 0.575 | 0.640 | 0.609 | -0.065 | 231.7 MB |
| pruning 30 % | 0.661 | 0.640 | 0.545 | +0.021 | 184.7 MB |
| all combined | 0.571 | 0.616 | 0.534 | -0.045 | 184.7 MB |

## What this resolves

- **Transfer works, partially.** A speech encoder does produce usable MER features, but on this
  data the ceiling is around F1 0.6 on three classes.
- **Unstructured pruning does not make inference faster.** Zeroed weights stay in dense matrices
  of unchanged shape and standard kernels do not skip them: 47.1 to 46.1 ms on CPU, 9.1 to 8.4 ms
  on GPU. Structured pruning or sparse kernels would be needed. What pruning *does* buy is size
  and, at 30 %, better generalisation.
- **float16 is the cheap win for the size constraint.** One call, half the raw footprint, no
  measurable quality loss. int8 dynamic quantisation failed: WavLM's custom attention indexes
  weight and bias tensors directly, and quantised layers no longer expose them as plain tensors.
- **Against a fair control, none of the regularisers helps.** The sweep's apparent jump at
  30 % sparsity (0.436 to 0.613) is largely the 7-epoch recovery fine-tuning, not the pruning:
  the 0 % row of stage 3 is measured *without* any fine-tuning, while every pruned row gets it.
  Stage 6 gives the control the same 7 epochs, and then pruning lands 0.064 F1 *below* it.
- **Weight decay at 1e-3 is inert here.** It does change the weights, but not one of the 86
  validation predictions. The confusion matrices of the control and the weight-decay run are
  identical. Over 490 steps the decoupled decay factor is ~0.9995.
- **The binding constraint is data, not architecture.** 366 windows from 2 patients, 83 %
  overlapping. Every technique here that helped, helped by fighting overfitting.

## Layout

```
const.py                 all configuration; edit hyperparameters here
core.py                  dataset, model, train/eval loops, measurement, pruning

01_prepare_data.py       raw CSV into 16 kHz wav windows plus dataset/manifest.csv
02_train_baseline.py     reference model; writes models/baseline.pt, metrics, curves
03_prune_sweep.py        sparsity sweep; writes pruned_metrics.json, pruned_curves.png
04_quant_fp16.py         fp32 vs fp16 at one sparsity; writes quant_fp16_metrics.json
05_augment_dropout.py    augmentation + dropout + pruning + fp16
06_ablation.py           each regularisation in isolation; writes ablation_metrics.json
```

Run in order. Each script writes a JSON file with its full config and results, so a table can be
rebuilt without retraining. Every file starts with a header comment stating its methodological
decisions.

## Running

```bash
pip install -r requirements.txt
python 01_prepare_data.py
python 02_train_baseline.py
python 03_prune_sweep.py
```

Stage 1 expects the raw annotated recordings in `brain_layer_annotations/<class>/<patient>/`,
obtained from the data DOI above; they are not in this repository. Stages 3 to 5 require
`models/baseline.pt` from stage 2.

## Known limitations

- **Single patient per split.** Any difference of a few F1 points between variants is
  indistinguishable from between-patient variance.
- **Model selection on the validation set.** `prune_and_recover` keeps the epoch with the best
  validation F1 and that same score is reported, which biases it upward. Treat it as an upper
  bound.
- **Only the head is checkpointed.** `02_train_baseline.py` does train the last two encoder
  layers (`N_UNFREEZE_LAYERS = 2`, 14.18 M parameters, lr 5e-5) but saves only `head_state`.
  Stages 3 to 6 therefore reload a pretrained backbone and pair it with a head that was trained
  against a different one. That mismatch is one reason the 0 % row of the sweep (F1 0.436) sits
  below the baseline's own 0.510; the other is that 0.510 is the best of 20 epochs while 0.436
  is a single measurement. The split between the two was not measured.
- **Differences are a handful of windows.** Validation is 86 windows, so one window is 0.012
  accuracy and the whole spread in the ablation table is 14 windows. Treat the ordering as
  weak evidence, not a ranking.
