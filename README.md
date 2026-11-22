# RDSG-YOLO: Reference-Difference Modeling and Structure-Guided Fusion for Insulator Defect Detection

Official PyTorch implementation of **RDSG-YOLO**, a lightweight and accurate insulator defect detector built upon YOLOv8s for UAV overhead transmission line inspection.

## Highlights

- **Reference-Difference Kernel Convolution (RDKC)**: Dynamically modulates grouped $3 \times 3$ local aggregation weights using center-to-neighborhood discrepancies sampled symmetrically across four orientations and two radii ($r \in \{2, 4\}$), suppressing cluttered outdoor backgrounds while highlighting abnormal shed structures (`rdsg_yolo/modules/rdkc.py`).
- **Structure-Guided Detail Fusion (SGDF)**: Incorporates backbone multi-scale difference maps ($E^3, E^4$) into a shared spatial protection gate that jointly regulates adaptive low-pass upsampling (ALPF) and local-similarity resampling offsets while enhancing fine shallow edges via AHPF (`rdsg_yolo/modules/sgdf.py`).
- **Alpha-CIoU Regression Loss**: Applies power-law weighting ($\kappa = 3$) to overlap, center-distance, and aspect-ratio terms to sharpen localization on compact insulator defects (`rdsg_yolo/losses/alpha_ciou.py`).

## Repository Structure

```text
RDSG-YOLO/
├── configs/
│   └── rdsg_yolov8s.yaml          # RDSG-YOLO architecture config
├── data/
│   ├── cplid.yaml                 # CPLID dataset config (insulator, defect)
│   └── idid.yaml                  # IDID dataset config (good/broken/flashover-shell)
├── rdsg_yolo/
│   ├── modules/
│   │   ├── common.py              # Conv, Bottleneck, C2f, SPPF, Detect
│   │   ├── rdkc.py                # RDKC and C2fRDKC blocks
│   │   └── sgdf.py                # SGDF cross-scale fusion module
│   ├── losses/
│   │   └── alpha_ciou.py          # Alpha-CIoU loss and Ultralytics hook
│   ├── models/
│   │   ├── rdsg_model.py          # Standalone PyTorch RDSG-YOLO model
│   │   └── ultralytics_adapter.py # Ultralytics module registration
│   └── utils/
│       ├── degradations.py        # Gaussian blur, brightness, noise, JPEG transforms
│       └── metrics.py             # Median e_c and 1 - IoU localization metrics
├── scripts/
│   └── prepare_splits.py          # Group-aware 7:1:2 dataset splitter
├── tests/
│   └── test_model.py              # Forward/backward and loss unit tests
├── train.py                       # Training entrypoint
├── eval_robustness.py             # Degradation and localization error evaluation
├── predict_vis.py                 # Detection and activation heatmap visualization
└── requirements.txt
```

## Installation

Tested with Ubuntu 22.04, Python 3.11, PyTorch 2.2.2, and CUDA 12.1.

```bash
git clone https://github.com/chenmeifeng11/RDSG-YOLO.git
cd RDSG-YOLO
pip install -r requirements.txt
```

Verify installation and model shapes:

```bash
python tests/test_model.py
```

## Datasets

Experiments are conducted on two public benchmarks:

1. **CPLID** (Chinese Power Line Insulator Dataset): 848 images (600 normal, 248 defective) with two classes (`insulator`, `defect`), split 7:1:2 into 594 train / 85 val / 169 test images.
2. **IDID** (Insulator Defect Image Dataset, EPRI V1.1): 1596 images evaluated across three shell categories (`good-shell`, `broken-shell`, `flashover-shell`), split 7:1:2 into 1116 train / 160 val / 320 test images.

Prepare group-aware 7:1:2 splits (keeping source images and derived copies in the same split):

```bash
python scripts/prepare_splits.py \
    --images /path/to/raw_images \
    --labels /path/to/yolo_labels \
    --out datasets/cplid
```

Update `path:` in `data/cplid.yaml` and `data/idid.yaml` before training.

## Training

Train on CPLID (200 epochs, $640 \times 640$, batch size 16, SGD, $\kappa = 3.0$):

```bash
python train.py --cfg configs/rdsg_yolov8s.yaml --data data/cplid.yaml --epochs 200 --batch 16 --kappa 3.0
```

Train on IDID (update `nc: 3` in `configs/rdsg_yolov8s.yaml`):

```bash
python train.py --cfg configs/rdsg_yolov8s.yaml --data data/idid.yaml --epochs 200 --batch 16 --kappa 3.0
```

## Evaluation & Visualization

Run localization precision ($e_c$, $1 - \mathrm{IoU}$) and progressive degradation benchmarks (Gaussian blur, brightness reduction, Gaussian noise, and JPEG compression):

```bash
python eval_robustness.py --weights runs/train/rdsg_yolo/weights/best.pt --data data/cplid.yaml
```

Generate detection results and activation heatmaps:

```bash
python predict_vis.py --weights runs/train/rdsg_yolo/weights/best.pt --source datasets/cplid/images/test --out runs/vis
```

