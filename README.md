# Cost-aware gating of deep edge detection for highly imbalanced industrial welding video

Code and evaluation artifacts for:

> Uddin, M. A., Rahman, S. S. M. M., & Gaber, J. (2026). *Cost-Aware Gating of Deep Edge Detection for Highly
> Imbalanced Industrial Welding Video.* Machine Learning with Applications (under review, MLWA-D-26-02065).

A cheap gate (frozen ImageNet backbone + linear or one-class head) decides which video frames are passed to
Holistically-Nested Edge Detection (HED). The speed-up follows the additive cost model `S = 1/(rho + q)`.
This repository contains everything needed to reproduce the evaluation **except the welding frames, which are
confidential** (provided by an industrial partner).

## Contents

| Path | What it is |
|---|---|
| `notebooks/MLWA_experiments_v4.ipynb` | The complete experiment pipeline (one notebook, runs on Google Colab or Kaggle). Produces every number, table and figure of the paper. |
| `src/paper_assets_v4.py` | Turns the notebook's CSV outputs into the LaTeX tables and number macros of the manuscript (also embedded in the notebook). |
| `release/fold_assignments.csv` | Recording-grouped 5-fold assignment of every frame for each of the 3 repetitions (seeds 0, 1, 2). |
| `release/per_fold_results.csv` | Per-fold metrics (PR-AUC, FPR at 100% recall, threshold, recall, FPR, routed share q) of all 60 gate configurations. |
| `release/oof_scores_*.csv` | Out-of-fold gate scores and selected thresholds of the selected gates (CPU-best, GPU-best, reference). |
| `release/hyperparameters.csv`, `release/config.json` | All settings used. |
| `results/tables/*.csv` | All result tables of the paper (cross-validation, nested selection, uncertainty, threshold rules, station calibration, perturbations, end-to-end timing in three settings, edge-map fidelity, prevalence). |
| `results/figures/*.png` | Figures that contain no frame images. |
| `results/environment.json` | Hardware and library versions of the reported run. |

## Main components of the notebook
- Frozen-feature gates (MobileNetV3-S/L, EfficientNet-B0, ResNet50 at two input sizes; HOG and image-statistics baselines) with six imbalance remedies (LR, weighted LR, LR + SMOTE, weighted focal loss, one-class Mahalanobis, one-class kNN).
- Recall-constrained threshold selection by inner recording-grouped cross-validation; split-conformal alternative.
- Recording-grouped CV (5 folds x 3 repetitions), nested evaluation of the complete model-selection procedure, recording-cluster bootstrap CIs, Clopper-Pearson recall bounds, corrected resampled t-tests.
- Latency measurement (interleaved rounds) and end-to-end validation of `S = 1/(rho + q)` in three settings: CPU, hybrid (gate on GPU, HED on CPU), GPU.
- A PyTorch port of the Caffe HED model (graph rebuilt from `deploy.prototxt`, weights read directly from the `.caffemodel`, verified against OpenCV).

## Running it with your own data
1. Frames: three folders `train_resized/`, `valid_resized/`, `test_resized/`, each with one sub-folder per class
   (the folder that comes second in alphabetical order is label 1 = joint), images 300 x 146 px (height x width).
   File names `img_<station>_<recording>_<frame>.jpg`; the recording is the grouping unit for all splits.
2. HED: `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` of Xie & Tu (2015), see https://github.com/s9xie/hed.
3. Colab: put both on Google Drive (`MyDrive/M2Project/...` and `MyDrive/Holistically-Nested_Edge_Detection/...`, or set
   `MLWA_BASE_DIR`, `MLWA_HED_PROTOTXT`, `MLWA_HED_MODEL`), GPU runtime, *Run all*.
   Kaggle: attach both as datasets, accelerator GPU T4 x2, *Save & Run All*. Paths are found automatically.
4. Long sections write checkpoints; re-running resumes. A full run takes about 2-3 h on a T4 runtime.

## Environment of the reported run
Python 3.13, TensorFlow 2.21, Keras 3.13, OpenCV 4.14 (CPU), NumPy 2.1, PyTorch (Colab default, CUDA), scikit-learn,
scikit-image, SciPy, pandas, matplotlib; Google Colab, 2-core Intel Xeon @ 2.00 GHz, NVIDIA Tesla T4.
See `results/environment.json` and `requirements.txt`.

## Data availability
The welding video frames are confidential and are not distributed. The fold assignments, per-fold results and
out-of-fold scores allow all reported statistics to be recomputed and checked without the images.

## License
Code: MIT (see `LICENSE`). Result files: CC BY 4.0.
