"""
Models compared:
  • YOLOv8  (Ultralytics, 2023)
  • YOLOv10 (Ultralytics, 2024)
  • YOLOv11 (Ultralytics, 2025)
Usage:
    python functions/inference.py \
        --data_root data/kitti_mots_yolo \
        --conf 0.25 \
        --max_images 200 \
        --wandb_project C5-Object-Detection-YOLO
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.wandb_logger import (
    Timer,
    finish,
    init_run,
    log_comparison_table,
    log_prediction_images,
)

# person=0, car=2
COCO_CLASSES_OF_INTEREST = {0: "person", 2: "car"}

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


def get_model_info(model: YOLO, weights_path: str) -> dict:
    """Extract parameter count and file size from a YOLO model."""
    num_params = sum(p.numel() for p in model.model.parameters())
    path = Path(weights_path) if Path(weights_path).exists() else None
    size_mb = path.stat().st_size / 1e6 if path and path.exists() else 0.0
    return {"num_params": num_params, "size_mb": size_mb}


def run_inference_single_model(
    model_name: str,
    weights: str,
    image_paths: list[Path],
    conf: float,
    iou: float,
    device: str,
    max_vis: int = 20,
) -> dict:
    """
    Run inference with one model on the image list.

    Returns a dict with timing and basic statistics.
    """
    print(f"Model : {model_name}  ({weights})")

    model = YOLO(weights)
    info = get_model_info(model, weights)

    _ = model(str(image_paths[0]), verbose=False)

    results_all = []
    total_time = 0.0
    total_boxes = {cls: 0 for cls in COCO_CLASSES_OF_INTEREST}

    for img_path in tqdm(image_paths, desc=f"Inference [{model_name}]"):
        t0 = time.perf_counter()
        res = model(
            str(img_path),
            conf=conf,
            iou=iou,
            classes=list(COCO_CLASSES_OF_INTEREST.keys()),  # filter at inference
            verbose=False,
            device=device,
        )
        t1 = time.perf_counter()
        total_time += (t1 - t0) * 1000  # ms

        results_all.append(res[0])

        # count predictions per class
        for box in res[0].boxes:
            cls_id = int(box.cls)
            if cls_id in total_boxes:
                total_boxes[cls_id] += 1

    avg_ms = total_time / len(image_paths)
    fps = 1000.0 / avg_ms

    summary = {
        "model": model_name,
        "weights": weights,
        "images": len(image_paths),
        "params (M)": round(info["num_params"] / 1e6, 2),
        "size (MB)": round(info["size_mb"], 2),
        "avg_ms/img": round(avg_ms, 2),
        "FPS": round(fps, 2),
        "det_person": total_boxes.get(0, 0),
        "det_car": total_boxes.get(2, 0),
        "det_total": sum(total_boxes.values()),
        "conf_threshold": conf,
    }

    print(
        f"{len(image_paths)} imgs | "
        f"{avg_ms:.1f} ms/img | {fps:.1f} FPS | "
        f"{summary['det_total']} detections"
    )

    return summary, results_all, image_paths


def main():
    parser = argparse.ArgumentParser(description="YOLO inference on KITTI-MOTS")
    parser.add_argument("--data_root", type=Path, default=Path("data/kitti_mots_yolo"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--max_images",
        type=int,
        default=200,
        help="Max images to run inference on (0=all)",
    )
    parser.add_argument(
        "--max_vis", type=int, default=30, help="Max images to upload as visualisations"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names from MODEL_ZOO to run. "
        "Default: all. Example: --models YOLOv8n YOLOv11s",
    )
    parser.add_argument(
        "--device", type=str, default="", help="cuda device, i.e. 0 or 0,1 or cpu"
    )
    parser.add_argument("--wandb_project", type=str, default="C5-Object-Detection-YOLO")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    img_dir = args.data_root / "images" / args.split
    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {img_dir}. " "Run the dataset converter first."
        )

    if args.max_images and args.max_images < len(image_paths):
        import random

        random.seed(42)
        image_paths = random.sample(image_paths, args.max_images)
    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} images in '{args.split}' split.")

    zoo = MODEL_ZOO
    if args.models:
        zoo = [(n, w) for n, w in MODEL_ZOO if n in args.models]

    comparison_rows = []

    for model_name, weights in zoo:
        # One WandB run per model so you can compare them in the dashboard
        run = init_run(
            run_name=f"inference_{model_name}",
            config={
                "model": model_name,
                "weights": weights,
                "task": "inference",
                "split": args.split,
                "conf": args.conf,
                "iou": args.iou,
                "num_images": len(image_paths),
                "device": device,
            },
            tags=["task_c", "inference", "pretrained", model_name, "KITTI-MOTS"],
            group="task_c_inference",
        )

        try:
            summary, results_all, paths_used = run_inference_single_model(
                model_name=model_name,
                weights=weights,
                image_paths=image_paths,
                conf=args.conf,
                iou=args.iou,
                device=device,
                max_vis=args.max_vis,
            )

            import wandb

            wandb.log(summary)

            # log prediction visualisations
            log_prediction_images(
                images_paths=paths_used,
                results_list=results_all,
                max_images=args.max_vis,
                prefix="predictions",
            )

            comparison_rows.append(summary)

        finally:
            import wandb

            wandb.finish()

    if len(comparison_rows) > 1:
        run = init_run(
            run_name="inference_comparison_summary",
            config={"task": "inference_summary"},
            tags=["task_c", "summary"],
            group="task_c_inference",
        )
        log_comparison_table(comparison_rows, table_name="inference_comparison")

        # also log bar-chart metrics for easy comparison
        import wandb

        for row in comparison_rows:
            wandb.log(
                {
                    f"fps/{row['model']}": row["FPS"],
                    f"params/{row['model']}": row["params (M)"],
                    f"det_total/{row['model']}": row["det_total"],
                }
            )
        finish()

    print("\nInference complete for all models.")
    print("Open your WandB project to explore results")


if __name__ == "__main__":
    main()
