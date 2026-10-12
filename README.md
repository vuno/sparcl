# SPARCL: Spatially Adaptive Regional Coverage for Active Learning

Official implementation by **VUNO Inc.**

PyTorch implementation of **SPARCL**, an active learning acquisition strategy for chest X-ray lesion segmentation
and detection that balances predictive uncertainty with *cumulative spatial coverage* of the image, together with
the baselines and the spatial reliability metrics (SI, PC) used in the paper.

## Method

Each active learning round selects a batch of `B` images from the unlabeled pool (Algorithm 1 of the paper):

1. **Uncertainty and its location.** For every unlabeled image `x`, the image uncertainty `U(x)` and the
   distribution `q(x)` of its uncertainty over the `M = K x L` grid bins.
   - Segmentation: pixel binary entropy `H(i)` of the predicted foreground probability,
     `U(x) = mean_i H(i)`, `h_m(x) = sum_{i in bin m} H(i)`.
   - Detection: `w_b = s_b * H(s_b)` for each predicted box `b` with score `s_b`, `U(x) = sum_b w_b`,
     `h_m(x) = sum_b w_b * |b ∩ bin m| / |b|`.
   - In both cases `q(x) = h(x) / sum_m h_m(x)`.
2. **Candidate set.** Only the top `r * B` images by `U(x)` are considered (`r = 5`); `U` is min-max normalized
   over the candidates.
3. **Positional-diversity gate.** `lambda = lambda_max * H(q_bar) / log M`, where `q_bar` is the mean of `q(x)`
   over the candidates. The gate closes, and SPARCL reduces to uncertainty sampling, only when the candidates'
   uncertainty concentrates in the same few bins.
4. **Cumulative coverage state.** `H_0 = B / |L+| * sum_{x in L+} g(x)`, where `g(x)` is the ground-truth lesion
   distribution over bins of each lesion-bearing labeled image (mask pixels for segmentation, box area for
   detection).
5. **Greedy batch.** Repeatedly add
   `argmax_x U(x) + lambda * sum_m [sqrt(H_m + q_m(x)) - sqrt(H_m)]` and update `H += q(x*)`. The objective is
   monotone submodular, so the greedy batch is a `(1 - 1/e)` approximation.

The implementation is in [`sparcl/acquisition/sparcl.py`](sparcl/acquisition/sparcl.py).

## Code structure

```
run_al.py                     Active learning loop: acquisition -> annotation -> retraining, per round
scripts/run_benchmark.sh      Strategies x seeds launcher with the paper settings
sparcl/
├── acquisition/
│   ├── sparcl.py             SPARCL (Algorithm 1): pool scoring, lesion distributions, gate, greedy batch
│   ├── simple.py             Random, Area-Random, Uncertainty, Area-Uncertainty, Diversity
│   ├── coreset.py            Core-Set k-center greedy
│   ├── taudis.py             TAUDIS
│   ├── usimc.py              USIMC
│   ├── loss_prediction.py    Learning Loss
│   └── paal.py               PAAL (accuracy predictor training and query)
├── data/                     SIIM-ACR, DDR, VinDr-CXR and ChestX-Det10 loaders and held-out splits
├── models/                   U-Net (EfficientNet-B4, scSE), Faster R-CNN (ResNet-50 FPN), PAAL predictor
├── metrics/                  Dice/IoU, Spatial Inconsistency, Performance Coverage, mAP/AP50/FROC
├── training.py               Per-round training with early stopping and held-out evaluation
├── uncertainty.py            Pool uncertainty used by the baseline strategies
└── losses.py                 Segmentation loss (Dice + BCE + Focal)
```

Each round, `run_al.py` loads the previous round's model, scores the unlabeled pool with the selected strategy,
adds the acquired images to the labeled set, trains a new model from the pretrained initialization on the labeled
set and evaluates it on the held-out set.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.5 and CUDA 12.1. `submodlib-py` is only needed by the USIMC baseline; without
it USIMC falls back to uncertainty sampling.

## Data

| Dataset | Task | Option | Expected layout |
|---|---|---|---|
| [SIIM-ACR](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | segmentation | `--siim_root` | `images/1024/dicom/<id>.png`, `images/1024/mask/<id>.png` |
| [DDR](https://github.com/nkicsl/DDR-dataset) | segmentation | `--ddr_root` | `{train,valid}/image/<id>.jpg`, `train/label/<lesion>/<id>.tif`, `valid/segmentation label/<lesion>/<id>.tif` |
| [VinDr-CXR](https://vindr.ai/datasets/cxr) | detection | `--vindr_root` | `dicom/{train,test}/<id>.dicom`, `old/annotations_{train,test}.csv` |
| [ChestX-Det10](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset) | detection | `--chestxdet10_root` | `train-old/<file>.png`, `test_data/<file>.png`, `train.json`, `test.json` |

Held-out sets: a stratified random split of 100 images for SIIM-ACR and DDR (`--num_validation_samples`), and the
official test split for VinDr-CXR and ChestX-Det10.

## Usage

Single run:

```bash
python run_al.py --dataset siim --siim_root /data/siim --siim_positive_only \
    --mode sparcl --grid 7 --num_samples 15 --round_num 25 --batch_size 8 --seed 0
```

All strategies and seeds of one benchmark, one job per GPU at a time:

```bash
DATASET=siim DATA_ROOT=/data/siim GPUS="0 1 2 3" SEEDS="0 1 2 3 4" bash scripts/run_benchmark.sh
```

`scripts/run_benchmark.sh` uses the paper settings:

| Dataset | Grid | B per round | Rounds | Batch size |
|---|---|---|---|---|
| SIIM-ACR (`--siim_positive_only`) | 7 x 7 | 15 | 25 | 8 |
| DDR (`--ddr_lesion MA`) | 7 x 7 | 15 | 24 | 8 |
| VinDr-CXR (`--target_lesion dispersed`) | 5 x 5 | 50 | 25 | 16 |
| ChestX-Det10 (`--target_lesion all`) | 5 x 5 | 50 | 50 | 16 |

A fully supervised reference run (random 80/20 split of the pool) is available with `--full_training`.

### Strategies (`--mode`)

| Mode | Strategy |
|---|---|
| `sparcl` | SPARCL (this work) |
| `random` | Uniform random sampling |
| `area_random` | Round-robin random sampling over `M` groups |
| `uncertainty` | Top-`B` uncertainty (mean of the top-10% pixel entropies; box-confidence score for detection) |
| `uncertainty_area` | Round-robin top uncertainty over `M` groups |
| `coreset` | Core-Set k-center greedy (Sener & Savarese, ICLR 2018) |
| `diversity` | Farthest-point sampling in feature space within the batch |
| `taudis` | TAUDIS (Yu et al., ICCV 2023) |
| `usimc` | USIMC (ALUNET, MIDL 2024) |
| `loss_prediction` | Learning Loss (Yoo & Kweon, CVPR 2019), segmentation only |
| `paal` | PAAL (Shi et al., IJCAI 2024), segmentation only |

### Main options

| Option | Default | Description |
|---|---|---|
| `--lambda_max` | 1.0 | Gate ceiling `lambda_max` |
| `--candidate_ratio` | 5 | `r`: SPARCL candidate set size is `r * B` |
| `--grid` | 7 | `K = L` of the spatial grid |
| `--train_epochs` / `--patience` | 150 / 10 | Training schedule and early stopping |
| `--init_seed` | 42 | Seed of the first-round selection |
| `--restore_best_weights` | off | Keep the best-epoch weights instead of the last-epoch weights |

## Outputs

Each run writes `<output_dir>/<dataset>/<mode>_seed<seed>_<timestamp>/` with `args.json`, per-round logs,
per-round checkpoints and `results.json`. Every round records the selected indices, SPARCL diagnostics
(`gate`, `lambda`, `h0`) and the held-out metrics of the best epoch:

- Segmentation: `dice`, `iou`, `si` (per-image Spatial Inconsistency over active bins, averaged over images with
  at least two active bins), `si_undefined_fraction`, `pc` (Performance Coverage against the previous round),
  pooled per-bin Dice statistics (`mean_bin_dice`, `worst_bin_dice`, ...).
- Detection: `mAP` (IoU 0.5:0.95), `AP50`, `AP75`, `froc_score` (mean sensitivity at 1/8-8 false positives per
  image), `auc_froc`.

## Citation

```bibtex
@article{sparcl,
  title  = {SPARCL: Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis},
  author = {},
  year   = {}
}
```

## License

Copyright 2026 VUNO Inc. All rights reserved.
