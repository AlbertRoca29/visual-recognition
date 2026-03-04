"""
Usage
-----
Train + eval:
python finetune_fasterrcnn_deart.py

Eval only:
python finetune_fasterrcnn_deart.py --skip_train
"""

import os, json, argparse, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
import albumentations as A
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = "/data/113-2/users/kpurkayastha/MCV/C5"
DEART_DIR        = os.path.join(BASE_DIR, "datasets/DeART")
DEART_IMG_DIR    = os.path.join(DEART_DIR, "images")
DEART_TRAIN_JSON = os.path.join(DEART_DIR, "deart_coco_train.json")
DEART_VAL_JSON   = os.path.join(DEART_DIR, "deart_coco_val.json")
KITTI_DIR        = os.path.join(BASE_DIR, "datasets/KITTI-MOTS")
KITTI_IMG_DIR    = os.path.join(KITTI_DIR, "training/image_02")
KITTI_INST_DIR   = os.path.join(KITTI_DIR, "instances_txt")
WEIGHTS_DIR      = os.path.join(BASE_DIR, "weights")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

CHECKPOINT      = os.path.join(WEIGHTS_DIR, "taskef_fasterrcnn_frozen_deart_coco_full.pth")
LOG_FILE        = os.path.join(LOGS_DIR,    "taskef_fasterrcnn_frozen_deart_coco_full_eval.log")
LOG_FILE_KITTI  = os.path.join(LOGS_DIR,    "taskef_fasterrcnn_frozen_deart_coco_full_kitti_eval.log")

# KITTI split & class mapping
VAL_SEQS       = [f"{i:04d}" for i in [2,6,7,8,10,13,14,16,18]]
KITTI_TO_COCO  = {1: 3, 2: 1}   # KITTI car→COCO 3, pedestrian→COCO 1
KITTI_COCO_IDS = [1, 3]
LOSS_CURVE = os.path.join(LOGS_DIR,    "taskef_fasterrcnn_frozen_deart_coco_full_loss.png")

# ── COCO-80 IDs (used to filter DeART annotations) ────────────────────────────
COCO_80_IDS = set([
    1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
    22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,
    46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,
    67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,
])

NUM_CLASSES = 91   # keep full COCO head

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = 50
PATIENCE   = 5


# ── Dataset ───────────────────────────────────────────────────────────────────

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


class DeARTDataset(Dataset):
    """DeART dataset filtered to COCO-overlap categories."""
    def __init__(self, json_path, transforms=None):
        self.transforms = transforms
        self.img_dir = DEART_IMG_DIR
        with open(json_path) as f:
            data = json.load(f)
        img_map  = {img["id"]: img for img in data["images"]}
        ann_map  = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] not in COCO_80_IDS:
                continue
            ann_map[ann["image_id"]].append(ann)
        self.samples = []
        for img_id, anns in ann_map.items():
            if not anns:
                continue
            img_info = img_map[img_id]
            path = os.path.join(DEART_IMG_DIR, img_info["file_name"])
            if not os.path.exists(path):
                continue
            boxes  = [[a["bbox"][0], a["bbox"][1],
                       a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                      for a in anns]
            labels = [a["category_id"] for a in anns]
            self.samples.append((path, boxes, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms:
            try:
                aug    = self.transforms(image=img, bboxes=boxes,
                                         category_ids=labels)
                img    = aug["image"]
                boxes  = list(aug["bboxes"])
                labels = list(aug["category_ids"])
            except Exception:
                pass

        img_t = TF.to_tensor(Image.fromarray(img))
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
    """Faster R-CNN with frozen backbone, original 91-class COCO head kept."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # Freeze backbone
    for p in model.backbone.body.parameters():
        p.requires_grad = False
    # IMPORTANT: do NOT replace box_predictor — keep the 91-class COCO head
    print("  Keeping original 91-class COCO predictor head (coco_full mode).")
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, device):
    train_ds = DeARTDataset(DEART_TRAIN_JSON, transforms=get_transform(True))
    val_ds   = DeARTDataset(DEART_VAL_JSON,   transforms=None)
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
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
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
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
            print(f"  ✓ Checkpoint → {CHECKPOINT}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Task f – FRCNN Frozen DeART coco_full – Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(LOSS_CURVE, dpi=130); plt.close()
    return CHECKPOINT


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(device, ckpt_path):
    print(f"\nEvaluating {ckpt_path} on DeART val …")
    model = build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    val_ds   = DeARTDataset(DEART_VAL_JSON, transforms=None)
    allowed  = sorted(COCO_80_IDS)

    # Build COCO GT from val_ds
    ann_id = 1
    gt_dict = {
        "images":     [],
        "annotations": [],
        "categories": [{"id": cid, "name": str(cid)} for cid in allowed],
    }
    for img_id, (img_path, boxes, labels) in enumerate(val_ds.samples):
        gt_dict["images"].append({"id": img_id, "file_name": os.path.basename(img_path)})
        for (x1, y1, x2, y2), cat_id in zip(boxes, labels):
            w, h = x2 - x1, y2 - y1
            gt_dict["annotations"].append(
                {"id": ann_id, "image_id": img_id, "category_id": cat_id,
                 "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); tmp_gt = f.name
    coco_gt = COCO(tmp_gt); os.unlink(tmp_gt)

    results = []
    for img_id, (img_path, _, _) in enumerate(tqdm(val_ds.samples, desc="Eval")):
        img_t = TF.to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        pred  = model(img_t)[0]
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            lbl = int(label); sc = float(score)
            if sc < 0.05 or lbl not in COCO_80_IDS:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({"image_id": img_id, "category_id": lbl,
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": sc})

    coco_dt = coco_gt.loadRes(results) if results else coco_gt
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = allowed
    ev.evaluate(); ev.accumulate(); ev.summarize()

    lines = [
        "Task f – Faster R-CNN (frozen, coco_full head) | DeART val",
        f"  mAP@50:95 = {ev.stats[0]:.4f}",
        f"  mAP@50    = {ev.stats[1]:.4f}",
        f"  mAP@75    = {ev.stats[2]:.4f}",
    ]
    print("\n".join(lines))
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Log → {LOG_FILE}")


# ── KITTI evaluation ──────────────────────────────────────────────────────────

def _rle_to_bbox(rle_str, h, w):
    rle = {"size": [h, w], "counts": rle_str.encode()}
    x, y, bw, bh = mask_utils.toBbox(rle)
    return float(x), float(y), float(x+bw), float(y+bh)

def _parse_kitti(txt_path):
    anns = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            fid   = int(parts[0])
            obj_id= int(parts[1])
            h, w  = int(parts[3]), int(parts[4])
            cls_k = obj_id // 1000
            if cls_k not in KITTI_TO_COCO: continue
            x1,y1,x2,y2 = _rle_to_bbox(parts[5], h, w)
            anns[fid].append((KITTI_TO_COCO[cls_k], x1,y1,x2,y2))
    return anns

@torch.no_grad()
def evaluate_on_kitti(device, ckpt_path):
    print(f"\nEvaluating {ckpt_path} on KITTI-MOTS val (cross-domain) …")
    model = build_model(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Collect all val samples
    samples = []  # (img_path, img_id, [(cat_id, x1,y1,x2,y2)])
    img_id  = 0
    for seq in VAL_SEQS:
        txt = os.path.join(KITTI_INST_DIR, f"{seq}.txt")
        if not os.path.exists(txt): continue
        ann    = _parse_kitti(txt)
        seq_dir= os.path.join(KITTI_IMG_DIR, seq)
        for p in sorted(Path(seq_dir).glob("*.png")):
            fid = int(p.stem)
            samples.append((str(p), img_id, ann.get(fid, [])))
            img_id += 1
    print(f"  Val frames: {len(samples)}")

    # Build COCO GT
    gt_dict = {"images": [], "annotations": [],
                "categories": [{"id":1,"name":"pedestrian"},{"id":3,"name":"car"}]}
    ann_id  = 1
    for img_path, iid, gt_boxes in samples:
        gt_dict["images"].append({"id": iid, "file_name": os.path.basename(img_path)})
        for cat_id, x1, y1, x2, y2 in gt_boxes:
            w, h = x2-x1, y2-y1
            if w>0 and h>0:
                gt_dict["annotations"].append(
                    {"id": ann_id, "image_id": iid, "category_id": cat_id,
                     "bbox": [x1,y1,w,h], "area": w*h, "iscrowd": 0})
                ann_id += 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f); tmp_gt = f.name
    coco_gt = COCO(tmp_gt); os.unlink(tmp_gt)

    # Inference
    results = []
    for img_path, iid, _ in tqdm(samples, desc="KITTI eval"):
        img_t = TF.to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        pred  = model(img_t)[0]
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            lbl = int(label); sc = float(score)
            if sc < 0.05 or lbl not in KITTI_COCO_IDS: continue
            x1,y1,x2,y2 = box.tolist()
            results.append({"image_id": iid, "category_id": lbl,
                            "bbox": [x1,y1,x2-x1,y2-y1], "score": sc})

    def _ev(cat_ids):
        coco_dt = coco_gt.loadRes(results) if results else coco_gt
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = cat_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return ev.stats

    ov  = _ev(KITTI_COCO_IDS)
    ped = _ev([1])
    car = _ev([3])

    lines = [
        "Task f – FRCNN frozen + coco_full | evaluated on KITTI-MOTS val",
        "="*60,
        f"  Overall  mAP@50:95={ov[0]:.4f}  mAP@50={ov[1]:.4f}  mAP@75={ov[2]:.4f}",
        f"  Person   mAP@50:95={ped[0]:.4f}  mAP@50={ped[1]:.4f}  mAP@75={ped[2]:.4f}",
        f"  Car      mAP@50:95={car[0]:.4f}  mAP@50={car[1]:.4f}  mAP@75={car[2]:.4f}",
    ]
    print("\n".join(lines))
    with open(LOG_FILE_KITTI, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Log → {LOG_FILE_KITTI}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task f – Faster R-CNN frozen + coco_full head on DeART"
    )
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--eval_kitti", action="store_true",
                        help="Evaluate checkpoint on KITTI-MOTS val (cross-domain)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")
    print("Mode: coco_full | Frozen backbone | Training on DeART ∩ COCO-80")

    ckpt = CHECKPOINT
    if not args.skip_train:
        ckpt = train(args, device)
    elif not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"No checkpoint at {CHECKPOINT}.")

    if args.eval_kitti:
        evaluate_on_kitti(device, ckpt)
    else:
        evaluate(device, ckpt)


if __name__ == "__main__":
    main()
