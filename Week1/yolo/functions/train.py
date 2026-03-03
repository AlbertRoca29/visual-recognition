"""
Augmentations tuned (all native YOLO):
  hsv_h, hsv_s, hsv_v  — colour jitter
  flipud / fliplr       — geometric flips
  mosaic               — 4-image mosaic (very effective for small objects)
  mixup                — image mixup regularisation
  scale / translate    — scale/shift augmentation
  degrees / shear      — rotation & shear

Usage:
    python functions/train.py \
        --data_root data/kitti_mots_yolo \
        --models YOLOv8n YOLOv10s YOLOv11m \
        --epochs 50 \
        --batch 16 \
        --imgsz 640 \
        --wandb_project C5-Object-Detection-YOLO
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.wandb_logger import init_run, finish

MODEL_ZOO = [
    ("YOLOv8n",  "yolov8n.pt"),
    ("YOLOv8s",  "yolov8s.pt"),
    ("YOLOv8m",  "yolov8m.pt"),
    ("YOLOv10n", "yolov10n.pt"),
    ("YOLOv10s", "yolov10s.pt"),
    ("YOLOv10m", "yolov10m.pt"),
    ("YOLOv11n", "yolo11n.pt"),
    ("YOLOv11s", "yolo11s.pt"),
    ("YOLOv11m", "yolo11m.pt"),
    ("YOLOv26m", "yolo26m.pt"),
    ("YOLOv26s", "yolo26s.pt"),
]

AUG_PRESETS = {
    "light": dict(
        hsv_h=0.015, hsv_s=0.3, hsv_v=0.2,
        degrees=0.0,  translate=0.1, scale=0.4,
        shear=0.0,    perspective=0.0,
        flipud=0.0,   fliplr=0.5,
        mosaic=0.8,   mixup=0.0,
        copy_paste=0.0,
    ),
    "medium": dict(
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0,  translate=0.1, scale=0.5,
        shear=2.0,    perspective=0.0,
        flipud=0.0,   fliplr=0.5,
        mosaic=1.0,   mixup=0.1,
        copy_paste=0.0,
    ),
    "heavy": dict(
        hsv_h=0.020, hsv_s=0.7, hsv_v=0.5,
        degrees=10.0, translate=0.2, scale=0.6,
        shear=5.0,    perspective=0.0005,
        flipud=0.0,   fliplr=0.5,
        mosaic=1.0,   mixup=0.2,
        copy_paste=0.1,
    ),
}


def get_dataset_yaml(data_root: Path) -> str:
    # Return path to dataset.yaml, creating it if missing
    yaml_path = data_root / "dataset.yaml"
    if not yaml_path.exists():
        cfg = {
            "path":  str(data_root.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc":    2,
            "names": ["car", "pedestrian"],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"Created dataset.yaml → {yaml_path}")
    return str(yaml_path)



def train_model(
    model_name: str,
    weights: str,
    dataset_yaml: str,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    lrf: float,
    aug_preset: str,
    device: str,
    out_dir: Path,
    freeze_backbone: bool = False,
) -> Path:
  
    aug_kwargs = AUG_PRESETS[aug_preset]

    cfg = {
        "model":           model_name,
        "weights":         weights,
        "epochs":          epochs,
        "batch":           batch,
        "imgsz":           imgsz,
        "lr0":             lr0,
        "lrf":             lrf,
        "aug_preset":      aug_preset,
        "freeze_backbone": freeze_backbone,
        "device":          device,
        **aug_kwargs,
    }

    run = init_run(
        run_name = f"train_{model_name}_{aug_preset}",
        config   = cfg,
        tags     = ["task_e", "finetune", "KITTI-MOTS", model_name, aug_preset],
        group    = "task_e_finetune",
        notes    = f"Fine-tuning {model_name} on KITTI-MOTS with {aug_preset} augmentation",
    )

    model    = YOLO(weights)
    run_name = f"{model_name}_{aug_preset}"

    freeze = None
    if freeze_backbone:
        freeze = 10
        print(f"Freezing first {freeze} layers (backbone)")

    results = model.train(
        data    = dataset_yaml,
        epochs  = epochs,
        batch   = batch,
        imgsz   = imgsz,
        lr0     = lr0,
        lrf     = lrf,
        device  = device,
        name    = run_name,
        project = str(out_dir),
        exist_ok = True,
        plots   = True,
        save    = True,
        freeze  = freeze,
        **aug_kwargs,
    )

    best_weights = Path(out_dir) / run_name / "weights" / "best.pt"

    import wandb
    if best_weights.exists():
        # log best model as a WandB Artifact
        artifact = wandb.Artifact(
            name  = f"{run_name}_best",
            type  = "model",
            description = f"Best weights for {model_name} fine-tuned on KITTI-MOTS",
        )
        artifact.add_file(str(best_weights))
        wandb.log_artifact(artifact)

    finish()
    return best_weights



def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on KITTI-MOTS")
    parser.add_argument("--data_root",       type=Path,  default=Path("data/kitti_mots_yolo"))
    parser.add_argument("--models",          nargs="+",  default=["YOLOv8n", "YOLOv10s", "YOLOv11m"],
                        help="Models to fine-tune (from MODEL_ZOO)")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch",           type=int,   default=16)
    parser.add_argument("--imgsz",           type=int,   default=640)
    parser.add_argument("--lr0",             type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--lrf",             type=float, default=0.01,
                        help="Final learning rate factor (lr0 * lrf)")
    parser.add_argument("--aug",             type=str,   default="medium",
                        choices=list(AUG_PRESETS.keys()),
                        help="Augmentation preset")
    parser.add_argument("--all_aug_presets", action="store_true",
                        help="Run all 3 augmentation presets (ablation study)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone layers during fine-tuning")
    parser.add_argument("--out_dir",         type=Path,  default=Path("runs/finetune"))
    parser.add_argument("--device",          type=str,   default="")
    parser.add_argument("--wandb_project",   type=str,   default="C5-Object-Detection-YOLO")
    parser.add_argument("--wandb_entity",    type=str,   default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Output : {args.out_dir}\n")

    # set WandB env vars for Ultralytics auto-detection
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    dataset_yaml = get_dataset_yaml(args.data_root)
    zoo          = {n: w for n, w in MODEL_ZOO}

    aug_presets  = list(AUG_PRESETS.keys()) if args.all_aug_presets else [args.aug]
    best_weights = {}   # {model_name: Path}

    for model_name in args.models:
        if model_name not in zoo:
            print(f"Unknown model '{model_name}'. Available: {list(zoo.keys())}")
            continue

        for aug_preset in aug_presets:
            print(f"Fine-tuning : {model_name}  (aug={aug_preset})")

            best = train_model(
                model_name      = model_name,
                weights         = zoo[model_name],
                dataset_yaml    = dataset_yaml,
                epochs          = args.epochs,
                batch           = args.batch,
                imgsz           = args.imgsz,
                lr0             = args.lr0,
                lrf             = args.lrf,
                aug_preset      = aug_preset,
                device          = device,
                out_dir         = args.out_dir,
                freeze_backbone = args.freeze_backbone,
            )

            key = f"{model_name}_{aug_preset}"
            best_weights[key] = best
            print(f"Best weights → {best}")

    print("\n\nAll fine-tuned models:")
    for k, v in best_weights.items():
        print(f"  {k:30s} → {v}")

    print("\nTraining complete. Run evaluate.py with --finetuned to get COCO metrics.\n")


if __name__ == "__main__":
    main()
