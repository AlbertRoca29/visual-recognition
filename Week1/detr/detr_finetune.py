# detr_finetune_utils.py
from pathlib import Path
from collections import defaultdict
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import DetrImageProcessor, DetrForObjectDetection, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils

from config import TRAIN_IMG_DIR, INSTANCES_DIR
from utils import map_label_to_class, load_annotations, SegmentedObject

# ---------------- Hyperparameters ----------------
MODEL_NAME = "facebook/detr-resnet-50"
CONF_THRESH = 0.7
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 10
# TRAIN_SPLIT = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRED_FILE = "detr_finetune_predictions.txt"

# ---------------- Dataset ----------------
class AugmentedImageDataset(Dataset):
    def __init__(self, image_paths, annotations, augment=True, label_map=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.label_map = label_map or {}

        if augment:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05,0.05), rotate=(-15,15), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ]
        else:
            transforms = []

        transforms += [
            ToTensorV2()
        ]

        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="coco", label_fields=["class_ids"], min_visibility=0.3)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))[:, :, ::-1]
        objs = self.annotations.get(int(Path(path).stem), [])

        h_img, w_img = img.shape[:2]
        bboxes, class_ids = [], []

        # Create ignore mask from class 10 objects
        ignore_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for obj in objs:
            if obj.class_id == 10:
                rle_mask = maskUtils.decode(obj.mask)
                # Resize mask to match image dimensions if needed
                if rle_mask.shape != (h_img, w_img):
                    rle_mask = cv2.resize(rle_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                ignore_mask = np.logical_or(ignore_mask, rle_mask).astype(np.uint8)

        # Process only classes 1 and 2, filter out objects overlapping with ignore region
        for obj in objs:
            if obj.class_id not in {1, 2}:
                continue

            x, y, w, h = map(float, maskUtils.toBbox(obj.mask))

            x = np.clip(x, 0, w_img)
            y = np.clip(y, 0, h_img)
            w = np.clip(w, 0, w_img - x)
            h = np.clip(h, 0, h_img - y)

            if w > 1 and h > 1:
                # Check overlap with ignore region
                bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                x_int, y_int = int(x), int(y)
                w_int, h_int = int(w), int(h)
                bbox_mask[y_int:y_int+h_int, x_int:x_int+w_int] = 1

                # Calculate overlap ratio
                overlap = np.sum(bbox_mask & ignore_mask)
                bbox_area = bbox_mask.sum()
                overlap_ratio = overlap / bbox_area if bbox_area > 0 else 0

                # Only keep objects with minimal overlap with ignore region (< 30%)
                if overlap_ratio < 0.3:
                    bboxes.append([x, y, w, h])
                    class_ids.append(int(obj.class_id))

        transformed = self.transform(
            image=img,
            bboxes=bboxes,
            class_ids=class_ids
        )

        img = transformed["image"]
        bboxes = transformed["bboxes"]
        class_ids = transformed["class_ids"]

        annotations = [
            {
                "bbox": bbox,
                "category_id": int(self.label_map.get(cid, cid)),
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0
            }
            for bbox, cid in zip(bboxes, class_ids)
        ]

        target = {
            "image_id": int(Path(path).stem),
            "annotations": annotations
        }

        return img, target, str(path)

# ---------------- Load sequences and annotations ----------------
images_by_seq = defaultdict(list)
annotations_by_seq = {}

for child in sorted(TRAIN_IMG_DIR.iterdir()):
    if child.is_dir():
        for f in sorted(child.iterdir()):
            if f.suffix.lower() == ".png":
                images_by_seq[child.name].append(f)
        inst_file = INSTANCES_DIR / f"{child.name}.txt"
        annotations_by_seq[child.name] = load_annotations(inst_file)

# ---------------- Train/Val split by sequence ----------------
# Use a fixed split by sequence as requested
TRAIN_SEQUENCES = [f"{i:04d}" for i in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
VAL_SEQUENCES = [f"{i:04d}" for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

train_paths, val_paths = [], []

for seq in TRAIN_SEQUENCES:
    imgs = images_by_seq.get(seq, [])
    if imgs:
        train_paths.extend([(p, seq) for p in imgs])

for seq in VAL_SEQUENCES:
    imgs = images_by_seq.get(seq, [])
    if imgs:
        val_paths.extend([(p, seq) for p in imgs])

train_annotations = {}
for s in TRAIN_SEQUENCES:
    train_annotations.update(annotations_by_seq.get(s, {}))

val_annotations = {}
for s in VAL_SEQUENCES:
    val_annotations.update(annotations_by_seq.get(s, {}))

# ---------------- Model (for label mapping) ----------------
# Load the pretrained processor/model early so we can map dataset class ids
# (1=car, 2=person) to the model's label ids.
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# Build a mapping from our dataset numeric class ids to the model's label ids
# e.g. dataset 1 (car) -> model label id for 'car'
model_labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
label_map = {}
for ds_id, name in [(1, "car"), (2, "person")]:
    matches = [k for k, v in model_labels.items() if name in v]
    if not matches:
        raise AssertionError(f"Model labels do not contain '{name}'")
    label_map[ds_id] = matches[0]

# ---------------- Datasets ----------------
train_dataset = AugmentedImageDataset(
    [p for p,_ in train_paths],
    train_annotations,
    augment=True,
    label_map=label_map
)
val_dataset = AugmentedImageDataset(
    [p for p,_ in val_paths],
    val_annotations,
    augment=False,
    label_map=label_map
)

def collate_fn(batch):
    return list(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# ---------------- Model ----------------
# Processor and model already loaded above to build `label_map`.

# ---------------- Optional: freeze backbone ----------------
for name, param in model.named_parameters():
    if "backbone" in name:
        param.requires_grad = False

model.train()


optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, targets, _ in tqdm(train_loader, desc="Training"):

        inputs = processor(images=imgs, annotations=targets, return_tensors="pt").to(DEVICE)

        # Move everything to GPU
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(DEVICE)
            elif isinstance(v, list):
                inputs[k] = [{kk: vv.to(DEVICE) if isinstance(vv, torch.Tensor) else vv for kk,vv in t.items()} for t in v]

        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    print(f"Loss: {loss.item():.4f}")

# ---------------- Evaluation ----------------
model.eval()
summary = {"processed": 0, "predicted_total": 0}

with open(PRED_FILE, "w") as pf:
    pf.write("# seq_name frame track_id class_id x y w h score\n")

    for batch_imgs, batch_targets, batch_paths in tqdm(val_loader, desc="Evaluating on validation"):
        batch_imgs = [img.to(DEVICE) for img in batch_imgs]
        sizes = [(img.shape[1], img.shape[2]) for img in batch_imgs]

        # Prepare inputs via processor
        inputs = processor(
            images=[(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8) for img in batch_imgs],
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(**inputs)

        target_sizes = torch.tensor(sizes, device=DEVICE)
        detections = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CONF_THRESH
        )

        for img_tensor, img_path, det in zip(batch_imgs, batch_paths, detections):
            for score, label, box in zip(det["scores"].cpu().numpy(), det["labels"].cpu().numpy(), det["boxes"].cpu().numpy()):
                x0, y0, x1, y1 = box
                x, y, w, h = float(x0), float(y0), float(x1-x0), float(y1-y0)
                label_name = model.config.id2label[int(label)]
                class_id = map_label_to_class(label_name)
                if class_id is None:
                    continue
                frame_id = int(Path(img_path).stem)
                summary["processed"] += 1
                summary["predicted_total"] += 1
                pf.write(f"{Path(img_path).parent.name} {frame_id} -1 {class_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {score:.4f}\n")

print(f"Processed {summary['processed']} validation images")
print(f"Total predicted boxes: {summary['predicted_total']}")
