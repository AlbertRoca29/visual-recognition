"""
Usage
-----
Train + eval:
python finetune_fasterrcnn_kitti.py

Eval only (uses existing checkpoint):
python finetune_fasterrcnn_kitti.py --skip_train
"""

import os, json, random, argparse, tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch, torchvision
import torchvision.transforms.functional as TF
import albumentations as A
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = "/data/113-2/users/kpurkayastha/MCV/C5"
KITTI_DIR      = os.path.join(BASE_DIR, "datasets/KITTI-MOTS")
KITTI_IMG_DIR  = os.path.join(KITTI_DIR, "training/image_02")
KITTI_INST_DIR = os.path.join(KITTI_DIR, "instances_txt")
WEIGHTS_DIR    = os.path.join(BASE_DIR, "weights")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

CHECKPOINT     = os.path.join(WEIGHTS_DIR, "taskef_fasterrcnn_frozen_kitti.pth")
LOG_FILE       = os.path.join(LOGS_DIR,    "taskef_fasterrcnn_frozen_kitti_eval.log")
LOSS_CURVE     = os.path.join(LOGS_DIR,    "taskef_fasterrcnn_frozen_kitti_loss.png")
RUN_NAME       = "taskef_fasterrcnn_frozen_kitti"

# ── KITTI split & class mapping ───────────────────────────────────────────────
TRAIN_SEQS     = [f"{i:04d}" for i in [0,1,3,4,5,9,11,12,15,17,19,20]]
VAL_SEQS       = [f"{i:04d}" for i in [2,6,7,8,10,13,14,16,18]]
KITTI_TO_COCO  = {1: 3, 2: 1}     # KITTI car→COCO 3, pedestrian→COCO 1
KITTI_COCO_IDS = [1, 3]
NUM_CLASSES    = 4                  # bg + person(1) + unused(2) + car(3)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = 50
PATIENCE   = 5


# ── Dataset ───────────────────────────────────────────────────────────────────

def rle_to_bbox(rle_str, h, w):
    rle = {"size": [h, w], "counts": rle_str.encode()}
    x, y, bw, bh = mask_utils.toBbox(rle)
    return float(x), float(y), float(x + bw), float(y + bh)


def parse_kitti_instances(txt_path):
    anns = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id           = int(parts[0])
            object_id          = int(parts[1])
            img_h, img_w       = int(parts[3]), int(parts[4])
            rle_str            = parts[5]
            class_kitti        = object_id // 1000
            if class_kitti not in KITTI_TO_COCO:
                continue
            x1, y1, x2, y2 = rle_to_bbox(rle_str, img_h, img_w)
            anns[frame_id].append((KITTI_TO_COCO[class_kitti], x1, y1, x2, y2))
    return anns


def get_transform(train=True):
    if not train:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=15, p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
                 rotate=(-10, 10), p=0.3),
    ], bbox_params=A.BboxParams(format="pascal_voc",
                                label_fields=["category_ids"],
                                min_visibility=0.2))


class KittiMOTSDataset(Dataset):
    def __init__(self, sequences, transforms=None):
        self.transforms = transforms
        self.samples = []
        for seq in sequences:
            txt = os.path.join(KITTI_INST_DIR, f"{seq}.txt")
            if not os.path.exists(txt):
                continue
            ann = parse_kitti_instances(txt)
            seq_dir = os.path.join(KITTI_IMG_DIR, seq)
            for img_path in sorted(Path(seq_dir).glob("*.png")):
                fid = int(img_path.stem)
                gt  = ann.get(fid, [])
                if gt:
                    self.samples.append((str(img_path), gt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        boxes  = [[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt]
        labels = [cat for cat, *_ in gt]

        if self.transforms:
            try:
                aug = self.transforms(image=img, bboxes=boxes,
                                      category_ids=labels)
                img    = aug["image"]
                boxes  = aug["bboxes"]
                labels = aug["category_ids"]
            except Exception:
                pass

        img_t  = TF.to_tensor(Image.fromarray(img))
        if not boxes:
            target = {"boxes":  torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros(0,       dtype=torch.int64)}
        else:
            target = {"boxes":  torch.tensor(boxes,  dtype=torch.float32),
                      "labels": torch.tensor(labels, dtype=torch.int64)}
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(device):
    """Faster R-CNN with frozen ResNet-50 backbone."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Freeze backbone (ResNet-50 body)
    for p in model.backbone.body.parameters():
        p.requires_grad = False

    # Replace prediction head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, device):
    train_ds = KittiMOTSDataset(TRAIN_SEQS, transforms=get_transform(True))
    val_ds   = KittiMOTSDataset(VAL_SEQS,   transforms=None)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, collate_fn=collate_fn)

    n_gpus = torch.cuda.device_count()
    model  = build_model(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    inner  = model.module if isinstance(model, torch.nn.DataParallel) else model
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.Adam(params, lr=LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3,
                                                         factor=0.5, verbose=True)

    train_losses, val_losses = [], []
    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        # ── train
        model.train()
        ep_loss = 0.0
        for imgs, targets in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} train"):
            imgs    = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses  = model(imgs, targets)
            loss    = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= len(train_dl)
        train_losses.append(ep_loss)

        # ── val loss
        inner.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs    = [i.to(device) for i in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                inner.train()
                losses  = inner(imgs, targets)
                v_loss += sum(losses.values()).item()
                inner.eval()
        v_loss /= max(len(val_dl), 1)
        val_losses.append(v_loss)

        print(f"Epoch {epoch:3d} | train={ep_loss:.4f}  val={v_loss:.4f}")
        sched.step(v_loss)

        if v_loss < best_val:
            best_val   = v_loss
            no_improve = 0
            torch.save(inner.state_dict(), CHECKPOINT)
            print(f"  ✓ Checkpoint saved → {CHECKPOINT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    # Loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Task e – FRCNN Frozen KITTI – Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(LOSS_CURVE, dpi=130); plt.close()
    return CHECKPOINT


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(device, ckpt_path):
    print(f"\nEvaluating {ckpt_path} on KITTI-MOTS val …")
    model = build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    val_ds = KittiMOTSDataset(VAL_SEQS, transforms=None)

    # Build COCO GT
    gt_dict = {"images": [], "annotations": [],
                "categories": [{"id": 1, "name": "pedestrian"},
                                {"id": 3, "name": "car"}]}
    ann_id = 1
    for img_id, (img_path, gt_boxes) in enumerate(val_ds.samples):
        gt_dict["images"].append({"id": img_id, "file_name": os.path.basename(img_path)})
        for cat_id, x1, y1, x2, y2 in gt_boxes:
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                gt_dict["annotations"].append(
                    {"id": ann_id, "image_id": img_id, "category_id": cat_id,
                     "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
                ann_id += 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); tmp_gt = f.name
    coco_gt = COCO(tmp_gt); os.unlink(tmp_gt)

    # Run inference
    results = []
    for img_id, (img_path, _) in enumerate(tqdm(val_ds.samples, desc="Eval")):
        img_t = TF.to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        pred  = model(img_t)[0]
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            lbl = int(label); sc = float(score)
            if sc < 0.05 or lbl not in KITTI_COCO_IDS:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({"image_id": img_id, "category_id": lbl,
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": sc})

    def _run_eval(cat_ids, label):
        coco_dt = coco_gt.loadRes(results) if results else coco_gt
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = cat_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return ev.stats

    overall = _run_eval(KITTI_COCO_IDS, "Overall")
    ped     = _run_eval([1], "Person")
    car     = _run_eval([3], "Car")

    lines = [
        f"Task e – Faster R-CNN (frozen backbone) | KITTI-MOTS val",
        f"{'='*55}",
        f"  Overall  mAP@50:95={overall[0]:.4f}  mAP@50={overall[1]:.4f}  mAP@75={overall[2]:.4f}",
        f"  Person   mAP@50:95={ped[0]:.4f}  mAP@50={ped[1]:.4f}  mAP@75={ped[2]:.4f}",
        f"  Car      mAP@50:95={car[0]:.4f}  mAP@50={car[1]:.4f}  mAP@75={car[2]:.4f}",
    ]
    print("\n".join(lines))
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Log → {LOG_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task e – Faster R-CNN frozen backbone on KITTI-MOTS"
    )
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    ckpt = CHECKPOINT
    if not args.skip_train:
        ckpt = train(args, device)
    elif not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}. Run without --skip_train first.")

    evaluate(device, ckpt)


if __name__ == "__main__":
    main()
