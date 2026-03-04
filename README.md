# Visual Recognition C5: Object Detection

Repository for the **C5 module** of the Master in Computer Vision (MCV) programme. It covers object detection on the **KITTI-MOTS** and **DeART** datasets using four model families: DETR, Faster R-CNN, RT-DETR, and YOLO.

---
---

## Datasets

| Dataset | Purpose |
|---|---|
| **KITTI-MOTS** | Primary training & evaluation dataset (cars + pedestrians) |
| **DeART** | Cross-domain fine-tuning dataset |

Expected dataset layout (configurable via path constants in each script):

```
<base>/datasets/
├── KITTI-MOTS/
│   ├── training/image_02/<seq>/
│   ├── testing/image_02/<seq>/
│   └── instances_txt/<seq>.txt
└── DeART/
    ├── images/
    ├── deart_coco_train.json
    └── deart_coco_val.json
```

**Train / Val split (KITTI-MOTS)**
- Train sequences: `0000 0001 0003 0004 0005 0009 0011 0012 0015 0017 0019 0020`
- Val sequences: `0002 0006 0007 0008 0010 0013 0014 0016 0018`

---

## Models

### DETR (`Week1/detr/`)
Transformer-based detector from Facebook (`facebook/detr-resnet-50` via HuggingFace).

```bash
# Zero-shot inference
python Week1/detr/detr.py

# Fine-tune (10 epochs, with Albumentations augmentation)
python Week1/detr/detr_finetune.py
```

### Faster R-CNN (`Week1/frcnn/`)
COCO-pretrained ResNet-50 FPN; frozen backbone, only RPN and RoI heads are updated.

```bash
# Train + eval on KITTI-MOTS
python Week1/frcnn/finetune_fasterrcnn_kitti.py

# Eval only (requires existing checkpoint)
python Week1/frcnn/finetune_fasterrcnn_kitti.py --skip_train

# Train on DeART, eval on KITTI-MOTS
python Week1/frcnn/finetune_fasterrcnn_deart.py
```

### RT-DETR (`Week1/rtdetr/`)
RT-DETR-L with HGNetV2 backbone

```bash
# Train + eval
python Week1/rtdetr/finetune_rtdetr_kitti.py

# Eval only
python Week1/rtdetr/finetune_rtdetr_kitti.py --skip_train
```

### YOLO (`Week1/yolo/`)
Supports YOLOv8, YOLOv10, and YOLOv11 (n/s/m variants). Includes light / medium / heavy augmentation presets and Weights & Biases logging.

```bash
# 1. Convert KITTI-MOTS to YOLO format
python Week1/yolo/data/kitti_mots_to_yolo.py

# 2. Run the full pipeline (convert → train → evaluate → infer → analyse)
python Week1/yolo/run_all.py --wandb_project C5-Object-Detection-YOLO

# Quick test (2 epochs, subset of images)
python Week1/yolo/run_all.py --quick_test

# Train specific models with a chosen augmentation preset
python Week1/yolo/functions/train.py \
    --data_root data/kitti_mots_yolo \
    --models YOLOv8n YOLOv11m \
    --epochs 50 --batch 16 --aug medium

# Evaluate
python Week1/yolo/functions/evaluate.py \
    --data_root data/kitti_mots_yolo \
    --models YOLOv8n YOLOv11m
```

---

## Model Analysis

Profile and compare latency, FPS, GPU memory, and parameter count across all four model families on KITTI-MOTS and DeART:

```bash
python Week1/model_analysis.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AlbertRoca29/visual-recognition.git
cd visual-recognition

# Install shared dependencies
pip install -r Week1/requirements.txt

# For YOLO experiments, also install
pip install -r Week1/yolo/requirements.txt
```

**Key dependencies**

| Package | Version | Used by |
|---|---|---|
| `torch` | ≥ 2.0 | All |
| `transformers` | ≥ 4.35 | DETR |
| `torchvision` | ≥ 0.15 | Faster R-CNN |
| `ultralytics` | ≥ 8.3 | YOLO, RT-DETR |
| `pycocotools` | ≥ 2.0.6 | All |
| `albumentations` | ≥ 1.3 | DETR, Faster R-CNN |
| `wandb` | ≥ 0.16 | YOLO |

---

## Evaluation

All models are evaluated using **COCO-style metrics** (mAP @ IoU 0.50:0.95) on the two target classes:

- **Car** (KITTI class 1 → COCO id 3)
- **Pedestrian** (KITTI class 2 → COCO id 1)

Results and training curves are logged to **Weights & Biases** (project `C5-Object-Detection-YOLO` by default for YOLO runs).
