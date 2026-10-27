<div align="center">

# SPARCL

### Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis

**Juho Jung** · **Seongjae Kang** · **Mijoo Kim** · **Sungduk Cho** · **Yunkyoung Jun** · **Sunghoon Joo** · **Yoona Hwang**<sup>†</sup>

VUNO Inc., Seoul, South Korea  
<sup>†</sup>Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2610.17808-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17808)
[![Paper](https://img.shields.io/badge/Paper-HTML-blue.svg)](https://arxiv.org/html/2610.17808)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

> **TL;DR** — Uncertainty-driven active learning keeps picking images whose uncertain regions overlap, which leaves
> parts of the lung annotated rarely or never. **SPARCL** adds a *cumulative spatial coverage* term with diminishing
> returns to the acquisition score, so each batch is both uncertain and spread across the image. With it, the
> model reaches **95% of fully supervised Dice with 24% of the labels**. The best baseline needs 29%.

## Highlights

- 🫁 **Two spatial regimes.** CXR pathologies are either *focal* or *diffuse*, and existing active learning
  methods do not perform consistently across both.
- 🎯 **Entropy-gated coverage acquisition.** SPARCL combines uncertainty with a concave coverage gain over a
  spatial grid. When the candidates show no spatial variation, a positional-diversity gate falls back to plain
  uncertainty sampling.
- 📐 **Near-optimal greedy batch.** The coverage term is a concave-over-modular function, so the objective is
  monotone submodular and linear-time greedy selection inherits the standard `(1 − 1/e)` approximation guarantee.
- 📊 **Spatial reliability metrics.** *Spatial Inconsistency* (SI) and *Performance Coverage* (PC) complement
  Dice by exposing regional failures that aggregate scores hide.
- 🔁 **Segmentation and detection.** Evaluated on SIIM-ACR, VinDr-CXR, ChestX-Det10, an in-house
  calcification cohort and, as a check outside CXR, DDR fundus images.

## Method

<!-- Overview figure: save Fig. 1 of the paper as assets/overview.png and uncomment the line below. -->
<!-- <p align="center"><img src="assets/overview.png" width="90%"></p> -->

Each active learning round selects a batch of `B` images from the unlabeled pool (Algorithm 1 of the paper):

1. **Uncertainty and its location.** For every unlabeled image `x`, compute the image uncertainty `U(x)` and the
   distribution `q(x)` of that uncertainty over the `M = K × L` grid bins.
   - *Segmentation:* pixel binary entropy `H(i)` of the predicted foreground probability,
     `U(x) = mean_i H(i)`, `h_m(x) = Σ_{i ∈ bin m} H(i)`.
   - *Detection:* `w_b = s_b · H(s_b)` for each predicted box `b` with score `s_b`, `U(x) = Σ_b w_b`,
     `h_m(x) = Σ_b w_b · |b ∩ bin m| / |b|`.
   - In both cases `q(x) = h(x) / Σ_m h_m(x)`.
2. **Candidate set.** Only the top `r · B` images by `U(x)` are considered (`r = 5`), and `U` is min-max
   normalized over the candidates.
3. **Positional-diversity gate.** `λ = λ_max · H(q̄) / log M`, where `q̄` is the mean of `q(x)` over the
   candidates. The gate closes, and SPARCL reduces to uncertainty sampling, only when the candidates'
   uncertainty concentrates in the same few bins.
4. **Cumulative coverage state.** `H_0 = B / |L⁺| · Σ_{x ∈ L⁺} g(x)`, where `g(x)` is the ground-truth lesion
   distribution over bins of each lesion-bearing labeled image (mask pixels for segmentation, box area for
   detection).
5. **Greedy batch.** Repeatedly add

   ```
   x* = argmax_x  U(x) + λ · Σ_m [ √(H_m + q_m(x)) − √H_m ]
   ```

   and update `H ← H + q(x*)`.

The implementation is in [`sparcl/acquisition/sparcl.py`](sparcl/acquisition/sparcl.py).

## Results

All numbers are mean ± std over 5 seeds (3 for DDR). **Budget@95%** is the fraction of the pool needed to reach
95% of fully supervised performance (lower is better). <sup>‡</sup>*Best baseline* is the strongest competing
method **in each column**, so a single row can mix methods. Several of the differences lie within one standard
deviation across seeds, so read them as consistent trends rather than significant gains. See the [paper](https://arxiv.org/abs/2610.17808) for
the full tables, including the ablations.

<details open>
<summary><b>Segmentation: label efficiency</b></summary>

| Dataset | Method | Dice@20% ↑ | AULC ↑ | Budget@95% ↓ |
|---|---|:-:|:-:|:-:|
| In-house calcification | Best baseline<sup>‡</sup> | 42.5 ± 1.2 | 43.7 ± 0.5 | 29.3 ± 0.9% |
| | **SPARCL** | **43.4 ± 1.3** | **45.1 ± 0.8** | **24.4 ± 1.4%** |
| SIIM-ACR | Best baseline<sup>‡</sup> | 54.1 ± 2.1 | 59.3 ± 2.3 | 22.4 ± 1.1% |
| | **SPARCL** | **55.1 ± 1.4** | **60.6 ± 1.7** | **21.2 ± 2.1%** |

</details>

<details open>
<summary><b>Spatial reliability at a 20% budget (in-house)</b></summary>

| Method | Global Dice ↑ | Mean Bin Dice ↑ | Dead-Bin % ↓ | SI ↓ | PC ↑ |
|---|:-:|:-:|:-:|:-:|:-:|
| Uncertainty | 39.8 ± 1.1 | 31.0 ± 2.0 | 16.0 ± 1.2 | 0.750 ± 0.045 | 0.111 ± 0.035 |
| **SPARCL** | **40.5 ± 1.1** | **36.1 ± 2.3** | **9.3 ± 1.2** | **0.571 ± 0.038** | **0.148 ± 0.009** |

Global Dice moves by less than one point, while bins with no correct prediction drop from 16% to 9%.

</details>

<details>
<summary><b>Detection at a 20% budget</b></summary>

| Dataset | Method | mAP ↑ | AP50 ↑ | FROC ↑ | Budget@95% ↓ |
|---|---|:-:|:-:|:-:|:-:|
| VinDr-CXR | Best baseline<sup>‡</sup> | 7.3 ± 0.2 | 20.7 ± 0.2 | 25.7 ± 0.7 | 42 ± 4% |
| | **SPARCL** | **7.7 ± 0.1** | **21.7 ± 0.8** | **27.1 ± 0.8** | **40 ± 3%** |
| ChestX-Det10 | Best baseline<sup>‡</sup> | 8.6 ± 0.2 | 26.1 ± 0.2 | 26.5 ± 0.3 | 73 ± 3% |
| | **SPARCL** | **8.9 ± 0.1** | **26.2 ± 0.5** | **26.6 ± 0.4** | **71 ± 2%** |

</details>

<details>
<summary><b>Beyond CXR: DDR fundus microaneurysm segmentation</b></summary>

| Method | Dice@20% ↑ | AULC ↑ | Final Dice ↑ |
|---|:-:|:-:|:-:|
| Best baseline<sup>‡</sup> | 23.4 ± 0.13 | 26.9 ± 0.13 | 32.2 ± 0.11 |
| **SPARCL** | **26.2 ± 0.23** | **29.5 ± 0.11** | **32.6 ± 0.15** |

</details>

## Getting Started

### Installation

```bash
git clone https://github.com/vuno/sparcl.git
cd sparcl
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.5 and CUDA 12.1. `submodlib-py` is only needed by the USIMC baseline; without
it, USIMC falls back to uncertainty sampling.

### Data

The in-house calcification cohort is not public. The four public benchmarks are supported out of the box:

| Dataset | Task | Option | Expected layout |
|---|---|---|---|
| [SIIM-ACR](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | segmentation | `--siim_root` | `images/1024/dicom/<id>.png`, `images/1024/mask/<id>.png` |
| [DDR](https://github.com/nkicsl/DDR-dataset) | segmentation | `--ddr_root` | `{train,valid}/image/<id>.jpg`, `train/label/<lesion>/<id>.tif`, `valid/segmentation label/<lesion>/<id>.tif` |
| [VinDr-CXR](https://vindr.ai/datasets/cxr) | detection | `--vindr_root` | `dicom/{train,test}/<id>.dicom`, `old/annotations_{train,test}.csv` |
| [ChestX-Det10](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset) | detection | `--chestxdet10_root` | `train-old/<file>.png`, `test_data/<file>.png`, `train.json`, `test.json` |

Held-out sets: a stratified random split of 100 images for SIIM-ACR and DDR (`--num_validation_samples`), and the
official test split for VinDr-CXR and ChestX-Det10.

### Quick start

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
|---|:-:|:-:|:-:|:-:|
| SIIM-ACR (`--siim_positive_only`) | 7 × 7 | 15 | 25 | 8 |
| DDR (`--ddr_lesion MA`) | 7 × 7 | 15 | 24 | 8 |
| VinDr-CXR (`--target_lesion dispersed`) | 5 × 5 | 50 | 25 | 16 |
| ChestX-Det10 (`--target_lesion all`) | 5 × 5 | 50 | 50 | 16 |

A fully supervised reference run (random 80/20 split of the pool) is available with `--full_training`.

### Strategies (`--mode`)

| Mode | Strategy |
|---|---|
| `sparcl` | **SPARCL (this work)** |
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
|---|:-:|---|
| `--lambda_max` | 1.0 | Gate ceiling `λ_max` |
| `--candidate_ratio` | 5 | `r`: SPARCL candidate set size is `r · B` |
| `--grid` | 7 | `K = L` of the spatial grid |
| `--train_epochs` / `--patience` | 150 / 10 | Training schedule and early stopping |
| `--init_seed` | 42 | Seed of the first-round selection |
| `--restore_best_weights` | off | Keep the best-epoch weights instead of the last-epoch weights |

### Outputs

Each run writes `<output_dir>/<dataset>/<mode>_seed<seed>_<timestamp>/` with `args.json`, per-round logs,
per-round checkpoints and `results.json`. Every round records the selected indices, SPARCL diagnostics
(`gate`, `lambda`, `h0`) and the held-out metrics of the best epoch:

- **Segmentation:** `dice`, `iou`, `si` (per-image Spatial Inconsistency over active bins, averaged over images
  with at least two active bins), `si_undefined_fraction`, `pc` (Performance Coverage against the previous
  round), and pooled per-bin Dice statistics (`mean_bin_dice`, `worst_bin_dice`, ...).
- **Detection:** `mAP` (IoU 0.5:0.95), `AP50`, `AP75`, `froc_score` (mean sensitivity at 1/8–8 false positives
  per image), `auc_froc`.

## Code Structure

Each round, `run_al.py` loads the previous round's model, scores the unlabeled pool with the selected strategy,
adds the acquired images to the labeled set, trains a new model from the pretrained initialization on the labeled
set and evaluates it on the held-out set.

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

## Citation

If you find SPARCL useful in your research, please cite:

```bibtex
@article{jung2026sparcl,
  title         = {SPARCL: Spatially Adaptive Regional Coverage for Active Learning in Chest X-ray Analysis},
  author        = {Jung, Juho and Kang, Seongjae and Kim, Mijoo and Cho, Sungduk and Jun, Yunkyoung and Joo, Sunghoon and Hwang, Yoona},
  journal       = {arXiv preprint arXiv:2610.17808},
  year          = {2026},
  eprint        = {2610.17808},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

Copyright 2026 VUNO Inc. All rights reserved.

This code is released for research reproducibility only. No license is granted for commercial use or redistribution.
