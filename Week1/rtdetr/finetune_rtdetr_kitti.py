"""

Usage
-----
Train + eval:
python finetune_rtdetr_kitti.py

Eval only (uses existing best.pt):
python finetune_rtdetr_kitti.py --skip_train
"""

import os, shutil, argparse, torch
from ultralytics import RTDETR

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "/data/113-2/users/kpurkayastha/MCV/C5"
KITTI_YAML = os.path.join(BASE_DIR,
             "datasets/KITTI-MOTS/yolo_correct/kitti_correct.yaml")
RTDETR_PT  = os.path.join(BASE_DIR, "rtdetr-l.pt")
WEIGHTS_DIR= os.path.join(BASE_DIR, "weights")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

RUN_NAME   = "taskh_rtdetr_frozen_kitti"
BEST_PT    = os.path.join(LOGS_DIR, RUN_NAME, "weights", "best.pt")
CKPT_COPY  = os.path.join(WEIGHTS_DIR, "taskh_rtdetr_r50_frozen_kitti.pt")
LOG_FILE   = os.path.join(LOGS_DIR, f"{RUN_NAME}_eval.log")

# ── Config ────────────────────────────────────────────────────────────────────
# RT-DETR-L architecture:
#   Layers 0-9  = HGNetV2 backbone (frozen via freeze=10)
#   Layers 10+  = transformer neck (AIFI, RepC3) + RTDETRDecoder head
FREEZE_LAYERS = 10
BATCH_SIZE    = 16
EPOCHS        = 10
LR            = 1e-4


def get_device():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    n = torch.cuda.device_count()
    return list(range(n)) if n > 0 else "cpu"


def train(args):
    if not os.path.exists(KITTI_YAML):
        raise FileNotFoundError(
            f"KITTI YAML not found: {KITTI_YAML}\n"
            "Run: python github/convert_kitti_yolo.py first."
        )
    print(f"\n{'='*60}")
    print(f"Fine-tuning RT-DETR-L | frozen backbone (layers 0-{FREEZE_LAYERS-1})")
    print(f"Dataset: KITTI-MOTS (nc=2: car, pedestrian)")
    print(f"{'='*60}")

    model = RTDETR(RTDETR_PT)
    model.train(
        data      = KITTI_YAML,
        epochs    = args.epochs,
        imgsz     = 640,
        batch     = args.batch_size,
        device    = get_device(),
        project   = LOGS_DIR,
        name      = RUN_NAME,
        exist_ok  = True,
        optimizer = "Adam",
        lr0       = args.lr,
        patience  = 5,
        seed      = 42,
        workers   = 8,
        amp       = False,
        mosaic    = 1.0,
        save      = True,
        freeze    = FREEZE_LAYERS,
    )

    if os.path.exists(BEST_PT):
        shutil.copy2(BEST_PT, CKPT_COPY)
        print(f"\nBest weights copied → {CKPT_COPY}")


def evaluate(args):
    ckpt = BEST_PT if os.path.exists(BEST_PT) else CKPT_COPY
    if not os.path.exists(ckpt):
        print(f"No checkpoint found at {ckpt}. Run without --skip_train first.")
        return

    print(f"\nEvaluating {ckpt} on KITTI-MOTS val …")
    model   = RTDETR(ckpt)
    metrics = model.val(
        data    = KITTI_YAML,
        imgsz   = 640,
        batch   = args.batch_size,
        device  = get_device(),
        verbose = True,
    )

    lines = [
        f"Task h – RT-DETR-L (frozen backbone) | KITTI-MOTS val",
        "=" * 55,
        f"mAP50-95 = {metrics.box.map:.4f}",
        f"mAP50    = {metrics.box.map50:.4f}",
        f"mAP75    = {metrics.box.map75:.4f}",
    ]
    print("\n".join(lines))
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Log → {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Task h – RT-DETR-L frozen backbone on KITTI-MOTS"
    )
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only evaluate existing checkpoint.")
    args = parser.parse_args()

    print(f"RT-DETR-L Frozen Backbone | KITTI-MOTS | epochs={args.epochs}")
    if not args.skip_train:
        train(args)
    evaluate(args)


if __name__ == "__main__":
    main()
