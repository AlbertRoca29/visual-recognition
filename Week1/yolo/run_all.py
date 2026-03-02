"""
Usage:
    # Run everything with defaults
    python run_all.py --wandb_project C5-Object-Detection-YOLO

    # Quick test on a few images and 2 epochs
    python run_all.py --quick_test

    # Only fine-tune + evaluate
    python run_all.py --steps 1e 1e2

    # Choose specific models
    python run_all.py --models YOLOv8n YOLOv11m
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], description: str):
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nCommand exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kitti_root", type=str, default="/ghome/group06/mcv/datasets/C5/KITTI-MOTS"
    )
    parser.add_argument("--data_root", type=str, default="data/kitti_mots_yolo")
    parser.add_argument("--out_dir", type=str, default="runs/finetune")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["YOLOv8n", "YOLOv8s", "YOLOv10n", "YOLOv10s", "YOLOv11n", "YOLOv11s"],
        help="Models to run (subset of MODEL_ZOO)",
    )

    VALID_STEPS = ["0", "1c", "1d", "1e", "1e2", "1g"]
    parser.add_argument(
        "--steps",
        nargs="+",
        default=VALID_STEPS,
        choices=VALID_STEPS,
        help="Pipeline steps to run",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument(
        "--aug", type=str, default="medium", choices=["light", "medium", "heavy"]
    )
    parser.add_argument(
        "--all_aug_presets",
        action="store_true",
        help="Train with all 3 aug presets (ablation)",
    )

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--max_images",
        type=int,
        default=500,
        help="Max images for inference/eval (0=all)",
    )
    parser.add_argument("--max_vis", type=int, default=30)

    parser.add_argument("--wandb_project", type=str, default="C5-Object-Detection-YOLO")
    parser.add_argument("--wandb_entity", type=str, default=None)

    parser.add_argument(
        "--device", type=str, default="", help="CUDA device, e.g. '0' or '0,1' or 'cpu'"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Use small subset (10 imgs, 2 epochs) for smoke test",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip robustness analysis in step 1g",
    )

    args = parser.parse_args()

    if args.quick_test:
        print("\nQUICK TEST: small subset, 2 epochs\n")
        args.max_images = 50
        args.epochs = 2
        args.batch = 4
        args.models = ["YOLOv8n"]

    py = sys.executable
    base = ["--wandb_project", args.wandb_project]
    if args.wandb_entity:
        base += ["--wandb_entity", args.wandb_entity]
    if args.device:
        base += ["--device", args.device]

    model_args = ["--models"] + args.models
    data_args = ["--data_root", args.data_root]
    infer_args = (
        ["--conf", str(args.conf), "--iou", str(args.iou)]
        + (["--max_images", str(args.max_images)] if args.max_images else [])
        + ["--max_vis", str(args.max_vis)]
    )

    steps = set(args.steps)
    rc_log = {}

    if "0" in steps:
        rc = run(
            [
                py,
                "data/kitti_mots_to_yolo.py",
                "--kitti_root",
                args.kitti_root,
                "--out_root",
                args.data_root,
            ],
            "Step 0 – Convert KITTI-MOTS → YOLO format",
        )
        rc_log["step_0_convert"] = rc

    if "1c" in steps:
        rc = run(
            [py, "functions/inference.py"] + data_args + model_args + infer_args + base,
            "Step 1c – Inference with pre-trained YOLO models",
        )
        rc_log["step_1c_inference"] = rc

    if "1d" in steps:
        rc = run(
            [py, "functions/evaluate.py"] + data_args + model_args + infer_args + base,
            "Step 1d – COCO evaluation of pre-trained models",
        )
        rc_log["step_1d_eval_pretrained"] = rc

    if "1e" in steps:
        train_extra = []
        if args.all_aug_presets:
            train_extra.append("--all_aug_presets")

        rc = run(
            [py, "functions/train.py"]
            + data_args
            + model_args
            + base
            + [
                "--epochs",
                str(args.epochs),
                "--batch",
                str(args.batch),
                "--imgsz",
                str(args.imgsz),
                "--lr0",
                str(args.lr0),
                "--aug",
                args.aug,
                "--out_dir",
                args.out_dir,
            ]
            + train_extra,
            "Step 1e – Fine-tune YOLO models on KITTI-MOTS",
        )
        rc_log["step_1e_finetune"] = rc

    if "1e2" in steps:
        weights_dir = Path(args.out_dir)
        rc = run(
            [py, "functions/evaluate.py"]
            + data_args
            + model_args
            + infer_args
            + base
            + ["--finetuned", "--weights_dir", str(weights_dir)],
            "Step 1e (follow-up) – COCO eval of fine-tuned models",
        )
        rc_log["step_1e2_eval_finetuned"] = rc

    if "1g" in steps:
        analysis_extra = []
        if args.skip_analysis:
            analysis_extra.append("--skip_robustness")

        rc = run(
            [py, "functions/analyze_models.py"]
            + data_args
            + model_args
            + base
            + [
                "--max_images",
                str(min(args.max_images or 100, 100)),
                "--conf",
                str(args.conf),
            ]
            + analysis_extra,
            "Step 1g – Analyse model differences (speed / params / robustness)",
        )
        rc_log["step_1g_analysis"] = rc

    print("Pipeline Summary")
    for step, rc in rc_log.items():
        status = "✅" if rc == 0 else "❌"
        print(f"  {status}  {step:35s}  (exit code {rc})")

    print(
        f"\n WandB project: https://wandb.ai/"
        f"{args.wandb_entity or 'entity'}/{args.wandb_project}"
    )


if __name__ == "__main__":
    main()
