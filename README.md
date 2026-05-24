# Hippocampus-Seg

[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025%20Accepted-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-3D%20Segmentation-EE4C2C)](#)

Official research code for hippocampus segmentation in CT volumes. This project
implements a 3D U-Net style segmentation framework with auxiliary edge
supervision and residual channel-attention modules for structure-aware
hippocampal mask prediction.

> Accepted to **MICCAI 2025**.

## Highlights

- 3D hippocampus segmentation from volumetric CT
- Center-crop based 3D preprocessing pipeline
- CT intensity windowing and min-max normalization
- Optional bone-response suppression
- Boundary map generation for auxiliary edge supervision
- Residual channel-attention segmentation model
- BCE + Dice optimization
- Dice, IoU, precision, recall, F1, and accuracy evaluation utilities

## Repository Structure

```text
.
├── main.py                 # Training entry point
├── eval.py                 # Evaluation and prediction export utilities
├── Options/
│   ├── BaseOptions.py      # Shared CLI arguments
│   ├── TrainOptions.py     # Training arguments
│   └── TestOptions.py      # Evaluation arguments
├── data/
│   ├── dataset.py          # CT and hippocampus mask preprocessing
│   └── load_data.py        # Train/test split generation
├── models/
│   ├── Unet.py             # Baseline 3D U-Net
│   ├── Unetedge.py         # 3D U-Net with edge branch
│   ├── Module.py           # Auxiliary modules
│   └── moduleRCA.py        # Residual channel-attention model
├── train/
│   └── trainer.py          # Training loop
└── utils/
    ├── loss.py             # BCE + Dice losses
    ├── seed.py             # Reproducibility helpers
    └── utils.py            # Boundary and post-processing utilities
```

## Method Summary

The default model in `models/moduleRCA.py` is a 3D encoder-decoder architecture
for binary hippocampus segmentation. It combines:

- residual feature paths for stable volumetric encoding
- channel attention blocks to recalibrate anatomical features
- a segmentation decoder and an auxiliary edge decoder
- boundary-guided loss terms to encourage sharper hippocampal contours

During training, the model predicts both the hippocampus mask and an edge map.
The trainer combines segmentation loss, edge loss, boundary consistency, and
inpainted-mask supervision.

## Data Layout

The current code expects a local HIPPO workspace. By default, paths are generated
under:

```text
D:\HIPPO\
```

Expected data organization:

```text
D:\HIPPO\
├── DATA\
│   ├── {case_id}\
│   │   ├── r*_CT.nii
│   │   └── lh+rh*.nii
│   └── ...
├── train_.xlsx
├── test_.xlsx
└── {date}\
    ├── model_parameters\
    └── test\
```

`data/load_data.py` generates `train_.xlsx` and `test_.xlsx` from the case
folders. Medical images and generated spreadsheets should remain local and are
not included in this repository.

## Preprocessing

For each case, the dataset performs:

1. Load CT and hippocampus mask with SimpleITK.
2. Reorient the volume using transpose and rotation operations.
3. Center crop to a default patch size of `96 x 128 x 128`.
4. Clip CT intensities with a configurable window.
5. Apply min-max normalization.
6. Suppress high-intensity bone responses.
7. Binarize hippocampus labels.
8. Generate a boundary map using Gaussian filtering and binary erosion.

Default training crop parameters:

```text
depth_crop_size = 96
crop_size       = 128
```

## Installation

Create a Python environment and install the core dependencies:

```bash
pip install torch numpy pandas SimpleITK scipy scikit-learn matplotlib openpyxl
```

The exact PyTorch installation command may depend on your CUDA version. See the
official PyTorch installation guide for CUDA-specific wheels.

## Training

Run training with the default residual channel-attention model:

```bash
python main.py \
  --device cuda \
  --date 0410 \
  --model Unet \
  --epochs 150 \
  --batch_size 4 \
  --lr 0.0001 \
  --edge 1 \
  --module 1 \
  --crop_size 128 \
  --depth_crop_size 96
```

Important options:

| Argument | Default | Description |
| --- | ---: | --- |
| `--device` | `cuda` | Training device |
| `--date` | `0410` | Experiment folder under `D:\HIPPO\` |
| `--model` | `Unet` | Model name used for checkpoint naming |
| `--epochs` | `150` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.0001` | Adam learning rate |
| `--edge` | `1` | Enable auxiliary edge supervision |
| `--module` | `1` | Enable module-based model branch |
| `--min_window` | `-20` | CT lower intensity window |
| `--max_window` | `100` | CT upper intensity window |
| `--crop_size` | `128` | In-plane crop size |
| `--depth_crop_size` | `96` | Depth crop size |

Model weights are saved to:

```text
D:\HIPPO\{date}\model_parameters\
```

## Evaluation

Evaluation utilities are implemented in `eval.py`. The evaluator loads a saved
model, creates test patches, predicts hippocampus masks, applies thresholding,
and exports NIfTI predictions.

Default metrics include:

- Dice
- IoU
- F1 score
- precision
- recall
- accuracy

Prediction outputs are written under:

```text
D:\HIPPO\{date}\test\
```

## Reproducibility

The training entry point calls `seed_everything` to set Python, NumPy, and PyTorch
random seeds. The default seed is:

```text
7136
```

## Notes

- The repository currently uses local Windows-style paths in several files.
  Update `D:\HIPPO\` paths if running on Linux, WSL, or a different storage
  layout.
- Raw medical images, generated Excel split files, checkpoints, and prediction
  outputs should not be committed to git.
- The public README will be updated with the final paper title, citation, and
  pretrained weights when they are ready for release.

## Citation

If this code is useful for your research, please cite our MICCAI 2025 paper.
The BibTeX entry will be added after the proceedings metadata is available.

```bibtex
@inproceedings{hippocampus_seg_2025,
  title     = {TBA},
  author    = {TBA},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2025}
}
```
