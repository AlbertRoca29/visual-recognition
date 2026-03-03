"""
Usage:
    python functions/evaluate.py \
        --data_root data/kitti_mots_yolo \
        --models YOLOv8n YOLOv10s YOLOv11m \
        --wandb_project C5-Object-Detection-YOLO
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.wandb_logger import finish, init_run, log_comparison_table

YOLO_CLS = {0: "car", 1: "pedestrian"}

COCO_TO_EVAL_CLS = {0: 1, 2: 0}  # coco_cls_id → eval_cls_id (0=car, 1=ped)

MODEL_ZOO = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv8m", "yolov8m.pt"),
    ("YOLOv10n", "yolov10n.pt"),
    ("YOLOv10s", "yolov10s.pt"),
    ("YOLOv10m", "yolov10m.pt"),
    ("YOLOv11n", "yolo11n.pt"),
    ("YOLOv11s", "yolo11s.pt"),
    ("YOLOv11m", "yolo11m.pt"),
]


def build_coco_gt(label_dir: Path, image_paths: list[Path]) -> dict:
    """
    Read YOLO-format label files and return a COCO-format GT dict.

    Returns:
        coco_gt_dict : dict compatible with pycocotools COCO constructor
        img_id_map   : {stem: image_id}  (str → int)
    """
    images = []
    anns = []
    ann_id = 1

    img_id_map = {}

    for img_id, img_path in enumerate(image_paths, start=1):
        lbl_path = label_dir / img_path.with_suffix(".txt").name

        # Read image dimensions
        from PIL import Image as PILImage

        img = PILImage.open(img_path)
        W, H = img.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": W,
                "height": H,
            }
        )
        img_id_map[img_path.stem] = img_id

        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # convert YOLO → COCO xywh (absolute pixels)
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                abs_w = bw * W
                abs_h = bh * H

                anns.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id + 1,  # COCO categories are 1-indexed
                        "bbox": [x1, y1, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    categories = [
        {"id": 1, "name": "car", "supercategory": "vehicle"},
        {"id": 2, "name": "pedestrian", "supercategory": "person"},
    ]

    return {
        "images": images,
        "annotations": anns,
        "categories": categories,
    }, img_id_map


def run_and_collect_preds(
    model_name: str,
    weights: str,
    image_paths: list[Path],
    img_id_map: dict,
    conf: float,
    iou: float,
    device: str,
    finetuned: bool = False,
) -> list[dict]:
    """
    Run YOLO inference and collect predictions in COCO format.

    For pre-trained (finetuned=False):
        model outputs COCO 80-class IDs  → remap via COCO_TO_EVAL_CLS
    For fine-tuned (finetuned=True):
        model outputs 0=car, 1=pedestrian  → map directly to category_id
    """
    model = YOLO(weights)

    preds = []
    for img_path in tqdm(image_paths, desc=f"  Predicting [{model_name}]"):
        img_id = img_id_map.get(img_path.stem)
        if img_id is None:
            continue

        # For pre-trained, restrict to person & car classes
        cls_filter = list(COCO_TO_EVAL_CLS.keys()) if not finetuned else None

        res = model(
            str(img_path),
            conf=conf,
            iou=iou,
            classes=cls_filter,
            verbose=False,
            device=device,
        )[0]

        if not finetuned:
            from PIL import Image as PILImage

            img = PILImage.open(img_path)
            W, H = img.size

        for box in res.boxes:
            coco_cls_id = int(box.cls)
            score = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if finetuned:
                cat_id = coco_cls_id + 1  # 0→1, 1→2
            else:
                # remap COCO class to eval class
                if coco_cls_id not in COCO_TO_EVAL_CLS:
                    continue
                cat_id = COCO_TO_EVAL_CLS[coco_cls_id] + 1  # 1-indexed

            preds.append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh
                    "score": score,
                }
            )

    return preds


def coco_evaluate(gt_dict: dict, preds: list[dict]) -> dict:
    """Run pycocotools COCOeval and return metric dict."""
    import tempfile, os

    # write to temp files (COCOeval needs files or dicts)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f)
        gt_path = f.name

    coco_gt = COCO(gt_path)
    os.unlink(gt_path)

    if not preds:
        print("  ⚠  No predictions – returning zeros.")
        return {
            k: 0.0
            for k in [
                "mAP",
                "mAP50",
                "mAP75",
                "mAP_s",
                "mAP_m",
                "mAP_l",
                "mAR_1",
                "mAR_10",
                "mAR_100",
            ]
        }

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "mAP": round(float(stats[0]), 4),
        "mAP50": round(float(stats[1]), 4),
        "mAP75": round(float(stats[2]), 4),
        "mAP_s": round(float(stats[3]), 4),
        "mAP_m": round(float(stats[4]), 4),
        "mAP_l": round(float(stats[5]), 4),
        "mAR_1": round(float(stats[6]), 4),
        "mAR_10": round(float(stats[7]), 4),
        "mAR_100": round(float(stats[8]), 4),
    }


def coco_evaluate_per_class(gt_dict: dict, preds: list[dict]) -> dict:
    """Run COCOeval per category and return per-class AP50."""
    import tempfile, os

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_dict, f)
        gt_path = f.name

    coco_gt = COCO(gt_path)
    os.unlink(gt_path)

    if not preds:
        return {}

    coco_dt = coco_gt.loadRes(preds)
    per_class = {}

    for cat in gt_dict["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]

        eval_obj = COCOeval(coco_gt, coco_dt, "bbox")
        eval_obj.params.catIds = [cat_id]
        eval_obj.params.iouThrs = np.array([0.50])
        eval_obj.evaluate()
        eval_obj.accumulate()

        # safe access: check if stats exists & has at least 1 element
        if hasattr(eval_obj, "stats") and len(eval_obj.stats) > 0:
            ap50 = float(eval_obj.stats[0])
        else:
            ap50 = 0.0

        per_class[f"AP50_{cat_name}"] = round(ap50, 4)

    return per_class


def main():
    parser = argparse.ArgumentParser(description="COCO evaluation of YOLO models")
    parser.add_argument("--data_root", type=Path, default=Path("data/kitti_mots_yolo"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--max_vis",
        type=int,
        default=0,
        help="Max number of visualizations (unused, for compatibility)",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max_images", type=int, default=0, help="0 = use all images")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Subset of MODEL_ZOO model names"
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use fine-tuned weights (different class mapping)",
    )
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=None,
        help="Directory containing fine-tuned .pt files " "(used when --finetuned)",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="C5-Object-Detection-YOLO")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    img_dir = args.data_root / "images" / args.split
    lbl_dir = args.data_root / "labels" / args.split

    image_paths = sorted(img_dir.glob("*.png"))
    if args.max_images:
        import random

        random.seed(42)
        image_paths = random.sample(image_paths, min(args.max_images, len(image_paths)))
        image_paths = sorted(image_paths)

    print(f"\n{len(image_paths)} images in '{args.split}' split.")

    print("Building COCO ground-truth …")
    gt_dict, img_id_map = build_coco_gt(lbl_dir, image_paths)
    total_gt_boxes = len(gt_dict["annotations"])
    print(f"GT boxes: {total_gt_boxes}")

    zoo = MODEL_ZOO
    if args.models:
        zoo = [(n, w) for n, w in MODEL_ZOO if n in args.models]

    comparison_rows = []

    for model_name, weights_default in zoo:
        if args.finetuned and args.weights_dir:
            w = args.weights_dir / f"{model_name}_medium" / "weights" / "best.pt"
            if not w.exists():
                print(f"Fine-tuned weights not found: {w}. Skipping.")
                continue
            weights = str(w)
            task_tag = "task_e"
        else:
            weights = weights_default
            task_tag = "task_d"

        run = init_run(
            run_name=f"eval_{model_name}_{'ft' if args.finetuned else 'pretrained'}",
            config={
                "model": model_name,
                "weights": weights,
                "finetuned": args.finetuned,
                "task": "evaluation",
                "split": args.split,
                "conf": args.conf,
                "iou": args.iou,
                "num_images": len(image_paths),
            },
            tags=[
                task_tag,
                "evaluation",
                model_name,
                "KITTI-MOTS",
                "finetuned" if args.finetuned else "pretrained",
            ],
            group=f"{task_tag}_evaluation",
        )

        try:
            preds = run_and_collect_preds(
                model_name=model_name,
                weights=weights,
                image_paths=image_paths,
                img_id_map=img_id_map,
                conf=args.conf,
                iou=args.iou,
                device=device,
                finetuned=args.finetuned,
            )

            print(f"\nRunning COCO evaluation …")
            metrics = coco_evaluate(gt_dict, preds)
            per_class = coco_evaluate_per_class(gt_dict, preds)

            all_metrics = {**metrics, **per_class}
            row = {"model": model_name, "finetuned": args.finetuned, **all_metrics}
            comparison_rows.append(row)

            import wandb

            wandb.log(all_metrics)

            print(f"\nResults for {model_name}:")
            for k, v in metrics.items():
                print(f"      {k:12s} : {v:.4f}")
            for k, v in per_class.items():
                print(f"      {k:20s} : {v:.4f}")

        finally:
            import wandb

            wandb.finish()

    if len(comparison_rows) > 1:
        run = init_run(
            run_name=f"eval_summary_{'ft' if args.finetuned else 'pretrained'}",
            config={"task": "evaluation_summary"},
            tags=[task_tag, "summary"],
            group=f"{task_tag}_evaluation",
        )
        log_comparison_table(comparison_rows, "evaluation_comparison")
        finish()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
